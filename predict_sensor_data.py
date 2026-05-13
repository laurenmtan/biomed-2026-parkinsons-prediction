"""
Predict Parkinson's Disease from sensor data using the Bayesian Neural Network model.

Usage:
    python predict_sensor_data.py <path_to_sensor_csv> [--model model.pth]
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# Model architecture (from parkinsons prediction bayesian neural network.py)
class ParkinsonBNN(nn.Module):
    def __init__(self, n_sensors=16, dropout=0.3):
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

        if mc_dropout:
            self.train()

        x = self.conv(x)
        x = self.fc(x)
        return x

    def predict_uncertainty(self, x, n_samples=100):
        """MC Dropout for uncertainty estimation"""
        self.eval()
        preds = torch.stack([self.forward(x, mc_dropout=True) for _ in range(n_samples)])
        return preds.mean(0), preds.std(0), preds


def load_sensor_data(csv_path):
    """Load and preprocess sensor data from CSV."""
    df = pd.read_csv(csv_path)

    # Get sensor columns (sensor_1 through sensor_16)
    sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
    sensor_cols = sorted(sensor_cols, key=lambda x: int(x.split('_')[1]))

    if not sensor_cols:
        raise ValueError("No sensor columns found in CSV file")

    print(f"Found {len(sensor_cols)} sensor columns: {sensor_cols[0]} to {sensor_cols[-1]}")
    print(f"Total timesteps: {len(df)}")

    # Extract sensor data
    X = df[sensor_cols].values  # Shape: (timesteps, n_sensors)

    return X, sensor_cols


def preprocess(X):
    """Normalize each sensor independently."""
    timesteps, n_sensors = X.shape
    X_norm = np.zeros_like(X, dtype=np.float32)

    for s in range(n_sensors):
        scaler = StandardScaler()
        sensor_data = X[:, s].reshape(-1, 1)
        X_norm[:, s] = scaler.fit_transform(sensor_data).flatten()

    X_norm = np.nan_to_num(X_norm, nan=0.0)
    return X_norm


def predict(model, X, device, n_mc_samples=100):
    """Make prediction with uncertainty estimation."""
    model.eval()

    # Add batch dimension: (timesteps, n_sensors) -> (1, timesteps, n_sensors)
    X_tensor = torch.FloatTensor(X).unsqueeze(0).to(device)

    with torch.no_grad():
        # Get prediction with uncertainty
        mean_pred, uncertainty, samples = model.predict_uncertainty(X_tensor, n_samples=n_mc_samples)

    prob_pd = mean_pred.cpu().numpy().flatten()[0]
    unc = uncertainty.cpu().numpy().flatten()[0]

    return prob_pd, unc


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_sensor_data.py <path_to_sensor_csv> [--model model.pth]")
        sys.exit(1)

    csv_path = sys.argv[1]
    model_path = None

    # Parse optional model path
    if '--model' in sys.argv:
        idx = sys.argv.index('--model')
        if idx + 1 < len(sys.argv):
            model_path = sys.argv[idx + 1]

    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nLoading sensor data from: {csv_path}")

    # Load data
    X, sensor_cols = load_sensor_data(csv_path)
    n_sensors = len(sensor_cols)

    # Preprocess
    X_norm = preprocess(X)

    # Initialize model
    model = ParkinsonBNN(n_sensors=n_sensors, dropout=0.3).to(device)

    # Try to load trained weights
    trained = False
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded trained model from: {model_path}")
            trained = True
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}")
    else:
        # Check default location
        default_model = os.path.join(os.path.dirname(csv_path), 'model.pth')
        if os.path.exists(default_model):
            try:
                model.load_state_dict(torch.load(default_model, map_location=device))
                print(f"Loaded trained model from: {default_model}")
                trained = True
            except Exception as e:
                print(f"Warning: Could not load model weights: {e}")

    if not trained:
        print("\n" + "="*60)
        print("WARNING: No trained model weights found!")
        print("The model is using random initialization.")
        print("Predictions are NOT meaningful without training.")
        print("="*60 + "\n")

    # Make prediction
    print("\nRunning prediction with MC Dropout (100 samples)...")
    prob_pd, uncertainty = predict(model, X_norm, device, n_mc_samples=100)

    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"P(Parkinson's Disease): {prob_pd:.1%}")
    print(f"P(Healthy Control):     {1-prob_pd:.1%}")
    print(f"Uncertainty (std):      {uncertainty:.3f}")
    print()

    if prob_pd > 0.5:
        classification = "PARKINSON'S DISEASE"
        confidence = prob_pd
    else:
        classification = "HEALTHY CONTROL"
        confidence = 1 - prob_pd

    print(f"Classification: {classification}")
    print(f"Confidence:     {confidence:.1%}")

    # Interpret uncertainty
    if uncertainty < 0.1:
        unc_level = "LOW (model is confident)"
    elif uncertainty < 0.2:
        unc_level = "MODERATE"
    else:
        unc_level = "HIGH (prediction unreliable)"
    print(f"Uncertainty:    {unc_level}")
    print("="*60)

    if not trained:
        print("\nNote: These results are from an UNTRAINED model.")
        print("To get meaningful predictions, train the model first using:")
        print("  python 'parkinsons prediction bayesian neural network.py'")

    return prob_pd, uncertainty


if __name__ == "__main__":
    main()
