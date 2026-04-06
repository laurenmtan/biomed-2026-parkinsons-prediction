import os
import pandas as pd

DATA_DIR = "C:/Users/Lauren/synapse partial data"

FOLDERS = {
    "Normal": 1,
    "Parkinsons": 0
}

PRESSURE_COLUMNS = [
    "LPressure1",
    "LPressure2",
    "LPressure3",
    "LPressure4",
    "LPressure5",
    "LPressure6",
    "LPressure7",
    "LPressure8",
    "LPressure9",
    "LPressure10",
    "LPressure11",
    "LPressure12",
    "LPressure13",
    "LPressure14",
    "LPressure15",
    "LPressure16",
    "RPressure1",
    "RPressure2",
    "RPressure3",
    "RPressure4",
    "RPressure5",
    "RPressure6",
    "RPressure7",
    "RPressure8",
    "RPressure9",
    "RPressure10",
    "RPressure11",
    "RPressure12",
    "RPressure13",
    "RPressure14",
    "RPressure15",
    "RPressure16"
]

def extract_mean_features(df):
    features = {}
    for col in PRESSURE_COLUMNS:
        if col in df.columns:
            features[f"mean_{col}"] = df[col].mean()
        else:
            features[f"mean_{col}"] = None
    return features


all_rows = []

for folder, label in FOLDERS.items():
    folder_path = os.path.join(DATA_DIR, folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)

            # Load Excel file
            #df = pd.read_csv(filepath)
            df = pd.read_csv(filepath, low_memory=False)

            # Select only pressure columns
            df_pressure = df[PRESSURE_COLUMNS]

            # Extract mean features
            feature_dict = extract_mean_features(df_pressure)

            # Add metadata
            feature_dict["file"] = filename
            feature_dict["label"] = label

            all_rows.append(feature_dict)

final_df = pd.DataFrame(all_rows)
final_df.to_csv("C:/Users/Lauren/Biomed 2026/synapse_pressure_mean_features.csv", index=False)