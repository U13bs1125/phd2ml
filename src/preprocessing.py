import pandas as pd
import numpy as np
import os

def preprocess_data(input_path, output_path="data/processed/2024pg.csv"):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dfg = pd.read_csv(input_path)

    # ---------------- DATES ----------------
    for col in ["Harvestdate", "Sowdate"]:
        if col in dfg.columns:
            dfg[col] = pd.to_datetime(dfg[col], errors="coerce")

    # Optional: convert to useful numeric features
    dfg["harvest_month"] = dfg["Harvestdate"].dt.month
    dfg["sow_month"] = dfg["Sowdate"].dt.month


    # ---------------- TARGETS ----------------
    dfg["Aflac"] = np.where(dfg["Afla"] > 10, 1, 0)
    dfg["Fumc"] = np.where(dfg["Fum"] > 4000, 1, 0)

    # ---------------- FEATURE GROUPS (BEFORE ENCODING) ----------------
    agro_cols = dfg.columns[:23].tolist()
    soil_cols = dfg.columns[23:40].tolist()
    weather_cols = dfg.columns[40:].tolist()

    # ---------------- CATEGORICAL ----------------
    categorical_cols = [
        "Region", "Crop", "Color", "Tillage",
        "Biocide", "Fertilizer", "Seedprep","Awareness",
        "Sowmethod", "Prevtime", "Prevcrop", "Country"
    ]

    categorical_cols = [c for c in categorical_cols if c in dfg.columns]

    dfg = pd.get_dummies(dfg, columns=categorical_cols, drop_first=True)

    #make the boolean columns as int
    bool_cols = dfg.select_dtypes(include=["bool"]).columns
    dfg[bool_cols] = dfg[bool_cols].astype(int)

    # ---------------- REBUILD FEATURE GROUPS ----------------
    def expand_cols(original_cols, df_columns):
        return [c for c in df_columns if any(orig in c for orig in original_cols)]

    agro_cols = expand_cols(agro_cols, dfg.columns)
    soil_cols = expand_cols(soil_cols, dfg.columns)
    weather_cols = expand_cols(weather_cols, dfg.columns)

    # Remove targets from feature groups
    targets = ["Aflac", "Fumc"]
    agro_cols = [c for c in agro_cols if c not in targets]
    soil_cols = [c for c in soil_cols if c not in targets]
    weather_cols = [c for c in weather_cols if c not in targets]

    # Save feature groups
    feature_dict = {
        "agro": agro_cols,
        "soil": soil_cols,
        "weather": weather_cols
    }

    # columns to drop (if they exist)
    drop_cols = ["Id", "Harvestdate", "Sowdate", "Longitude", "Latitude"]
    drop_cols = [col for col in drop_cols if col in dfg.columns]
    dfg.drop(columns=drop_cols, inplace=True)

    # Save processed data
    dfg.to_csv(output_path, index=False)

    print(f"✅ Processed data saved to: {output_path}")
    print(f"Shape: {dfg.shape}")

    return dfg, feature_dict
    

    

    