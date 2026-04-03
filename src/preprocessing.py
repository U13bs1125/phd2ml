import pandas as pd
import numpy as np
import os
import json


def preprocess_data(
    input_path,
    output_path="data/processed/2024pg.csv",
    artifact_path="artifacts"
):

    # ---------------- CREATE DIRECTORIES ----------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(f"{artifact_path}/features", exist_ok=True)
    

    # ---------------- LOAD ----------------
    dfg = pd.read_csv(input_path)
    dfg2 = pd.read_csv("data/selected/2024pg_sfm.csv")
    dfg2columns = dfg2.columns

    # ---------------- DATES ----------------
    for col in ["Harvestdate", "Sowdate"]:
        if col in dfg.columns:
            dfg[col] = pd.to_datetime(dfg[col], errors="coerce")

    dfg["harvest_month"] = dfg["Harvestdate"].dt.month
    dfg["sow_month"] = dfg["Sowdate"].dt.month

    # ---------------- TARGETS ----------------
    dfg["Aflac"] = np.where(dfg["Afla"] > 10, 1, 0)
    dfg["Fumc"] = np.where(dfg["Fum"] > 4000, 1, 0)

    targetscol = ["Aflac", "Fumc", "Afla", "Fum"]

    # ---------------- DEFINE FEATURE GROUPS ----------------

    # 🌱 SOIL FEATURES
    soil_properties = [
        "ph", "carbon_organic", "carbon_total", "nitrogen_total",
        "cation_exchange_capacity", "phosphorous_extractable",
        "potassium_extractable", "calcium_extractable",
        "magnesium_extractable", "iron_extractable",
        "zinc_extractable", "sulphur_extractable",
        "sand_content", "silt_content", "clay_content",
        "stone_content", "bulk_density",
    ]

    # 🌦 WEATHER FEATURES
    weather_prefixes = ["T2M", "RH2M", "PRECTOT", "ALLSKY"]

    # ---------------- CATEGORICAL ----------------
    categorical_cols = [
        "Color", "Tillage",
        "Biocide", "Fertilizer", "Seedprep", "Awareness",
        "Sowmethod", "Prevtime", "Prevcrop"
    ]

    categorical_cols = [c for c in categorical_cols if c in dfg.columns]

    dfg = pd.get_dummies(dfg, columns=categorical_cols, drop_first=True)

    # Convert bool → int
    bool_cols = dfg.select_dtypes(include=["bool"]).columns
    dfg[bool_cols] = dfg[bool_cols].astype(int)

    # ---------------- BUILD FEATURE GROUPS ----------------

    # Soil
    soil_cols = [c for c in dfg.columns if c in soil_properties]

    # Weather
    weather_cols = [
        c for c in dfg.columns
        if any(c.startswith(prefix) for prefix in weather_prefixes)
    ]

    # Agronomic = everything else except targets
    excluded = set(soil_cols + weather_cols + targetscol)

    agro_cols = [
        c for c in dfg.columns
        if c not in excluded
    ]

    # ---------------- CLEAN FEATURE GROUPS ----------------
    soil_cols = [c for c in soil_cols if c in dfg.columns and c in dfg2columns]
    weather_cols = [c for c in weather_cols if c in dfg.columns and c in dfg2columns]
    agro_cols = [c for c in agro_cols if c in dfg.columns and c in dfg2columns]

    # ---------------- FEATURE DICTIONARY ----------------
    feature_dict = {
        "soil": soil_cols,
        "weather": weather_cols,
        "agro": agro_cols,
        "weather_soil": weather_cols + soil_cols,
        "weather_soil_agro": agro_cols + soil_cols + weather_cols
    }

    # ---------------- DROP UNUSED ----------------
    drop_cols = ["Id", "Harvestdate", "Sowdate", "Longitude", "Latitude", "Country", "Region","Crop"]
    drop_cols = [col for col in drop_cols if col in dfg.columns]
    dfg.drop(columns=drop_cols, inplace=True)

    # ---------------- SAVE DATA ----------------
    dfg.to_csv(output_path, index=False)

    # ---------------- SAVE FEATURE DICT ----------------
    with open(f"{artifact_path}/features/feature_dict.json", "w") as f:
        json.dump(feature_dict, f, indent=4)

    # save feature groups separately for easy loading in SHAP analysis
    # SAVE EACH FEATURE SET (important for SHAP)
    for name, cols in feature_dict.items():
        with open(f"{artifact_path}/features/{name}.json", "w") as f:
            json.dump(cols, f)

    # ---------------- DEBUG PRINT ----------------
    print(f"\n✅ Processed data saved to: {output_path}")
    print(f"📁 Artifacts saved to: {artifact_path}/")

    print(f"\nShape: {dfg.shape}")

    print("\n📊 FEATURE SET SIZES:")
    print(f"🌱 Soil ({len(soil_cols)}): {soil_cols}")
    print(f"🌦 Weather ({len(weather_cols)}): {weather_cols}")
    print(f"🚜 Agronomic ({len(agro_cols)}): {agro_cols}")
    print(f"🌦+🌱 Weather+Soil: {len(feature_dict['weather_soil'])}")
    print(f"🌍 All: {len(feature_dict['weather_soil_agro'])}")

    return dfg, feature_dict