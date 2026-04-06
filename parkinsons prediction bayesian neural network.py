import numpy as np
import pandas as pd
import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#model
class ParkinsonBNN(nn.Module):
    def __init__(self, n_sensors=8, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        # Simple 3-layer CNN
        self.conv = nn.Sequential(
            nn.Conv1d(n_sensors, 32, 7, padding=3),
            nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x, mc_dropout=False):
        x = x.transpose(1, 2)  # (batch, time, sensors) -> (batch, sensors, time)

        # Apply dropout even in eval mode if mc_dropout=True
        if mc_dropout:
            self.train()

        x = self.conv(x)
        x = self.fc(x)
        return x

    def predict_uncertainty(self, x, n_samples=100):
        """MC Dropout for uncertainty"""
        self.eval()
        preds = torch.stack([self.forward(x, mc_dropout=True) for _ in range(n_samples)])
        return preds.mean(0), preds.std(0), preds


#training
def train_model(model, train_loader, val_loader, epochs=20, lr=0.001):
    """Simple training loop with early stopping"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    best_loss = float('inf')
    patience = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device).float().unsqueeze(1)
                val_loss += criterion(model(X), y).item()
        val_loss /= len(val_loader)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            best_state = model.state_dict()
        else:
            patience += 1
            if patience >= 15:
                model.load_state_dict(best_state)
                print(f"Early stopped at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: val_loss={val_loss:.4f}")

    return model


#eval
def evaluate(model, test_loader, n_mc=100):
    """Evaluate with uncertainty"""
    model.eval()
    all_preds, all_uncs, all_labels = [], [], []

    print(f"Evaluating with {n_mc} MC samples...")
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            mean, unc, _ = model.predict_uncertainty(X, n_samples=n_mc)
            all_preds.extend(mean.cpu().numpy().flatten())
            all_uncs.extend(unc.cpu().numpy().flatten())
            all_labels.extend(y.numpy())

    preds = np.array(all_preds)
    uncs = np.array(all_uncs)
    labels = np.array(all_labels)

    # Metrics
    binary = (preds > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(labels, binary, labels=[0,1], target_names=['Healthy', 'Parkinsons']))
    print(f"ROC-AUC: {roc_auc_score(labels, preds):.3f}")

    # Uncertainty analysis
    print(f"\nUncertainty - Mean: {uncs.mean():.3f}, Std: {uncs.std():.3f}")

    # Most uncertain cases
    top_uncertain = np.argsort(uncs)[-3:]
    print("\nMost uncertain predictions:")
    for i in top_uncertain:
        print(f"  P(PD)={preds[i]:.2f}±{uncs[i]:.2f} | True={labels[i]}")

    return {'predictions': preds, 'uncertainties': uncs, 'labels': labels}


#data preprocessing
def preprocess(X, y):
    """Normalize each sensor independently"""
    n_samples, timesteps, n_sensors = X.shape
    X_norm = np.zeros_like(X)

    for s in range(n_sensors):
        scaler = StandardScaler()
        sensor_data = X[:, :, s].reshape(-1, 1)
        X_norm[:, :, s] = scaler.fit_transform(sensor_data).reshape(n_samples, timesteps)

    X_norm = np.nan_to_num(X_norm, nan=0.0)
    return X_norm, y


if __name__ == "__main__":
    # dataloading
    X = []
    y = []

    pd_data_dir = 'C:/Users/Lauren/Biomed 2026/parkinsons' #insert personal filepath here
    pd_csv_files = glob.glob(os.path.join(pd_data_dir, '**', '*.csv'), recursive=True)

    for f in pd_csv_files:
        df = pd.read_csv(f)

        pressure_cols = [
            col for col in df.columns
            if col.lower().startswith("lpressure") or col.lower().startswith("rpressure")
        ]
        if pressure_cols:
            X.append(df[pressure_cols].values)
            y.append(1)  #pd label

    control_data_dir = 'C:/Users/Lauren/Biomed 2026/control' #insert personal filepath here
    control_csv_files = glob.glob(os.path.join(control_data_dir, '**', '*.csv'), recursive=True)

    for f in control_csv_files:
        df = pd.read_csv(f)

        pressure_cols = [
            col for col in df.columns
            if col.lower().startswith("lpressure") or col.lower().startswith("rpressure")
        ]
        if pressure_cols:
            X.append(df[pressure_cols].values)
            y.append(0)  #control label

    max_len = max(arr.shape[0] for arr in X)
    n_sensors = X[0].shape[1]

    X_padded = np.zeros((len(X), max_len, n_sensors))

    for i, arr in enumerate(X):
        L = arr.shape[0]
        X_padded[i, :L, :] = arr

    # Convert to numpy
    X = X_padded  # Shape: (n_samples, timesteps, n_sensors)
    y = np.array(y)  # Shape: (n_samples,)

    print(f"Data: X={X.shape}, y={y.shape}")

    # Preprocess
    X, y = preprocess(X, y)

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)

    # DataLoaders
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
                              batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
                            batch_size=32)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
                             batch_size=32)

    print("Train:", np.bincount(y_train))
    print("Val:", np.bincount(y_val))
    print("Test:", np.bincount(y_test))

    # Train
    model = ParkinsonBNN(n_sensors=X.shape[2], dropout=0.3).to(device)
    print(f"\nTraining on {device}...")
    model = train_model(model, train_loader, val_loader, epochs=100)

    # Evaluate
    results = evaluate(model, test_loader, n_mc=100)

    # Save
    torch.save(model.state_dict(), 'model.pth')
    print("\nModel saved to 'model.pth'")