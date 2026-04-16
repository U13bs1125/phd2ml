import pandas as pd
import numpy as np
import json

# ---------------- CONFIG ----------------
latgrid = 0.5
longgrid = 0.5

#load soil.jason and make items as list
with open("artifacts/features/soil.json", "r") as f:
    soil_items = json.load(f)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/preprocessed/2024pg.csv")
agrocols= ["Ota", "Zen", "Don", "Cultdays", "harvest_month", "sow_month", "Color_multicolor", "Color_red", "Color_white", "Color_yellow", "Tillage_notillage", "Tillage_shallowtillage", "Biocide_no", "Biocide_pesticide", "Fertilizer_notapplied", "Seedprep_notapplied", "Awareness_notaware", "Sowmethod_disperse", "Prevtime_lastseason", "Prevtime_lastyear", "Prevcrop_cottonflower", "Prevcrop_legume", "Prevcrop_mix", "Prevcrop_tuber"]


# rename for safety
df = df.rename(columns={"Latitude": "lat", "Longitude": "lon"})

# ---------------- CREATE GRID ----------------
df["lat_grid"] = (df["lat"] // latgrid) * latgrid
df["lon_grid"] = (df["lon"] // longgrid) * longgrid

df["lat_grid"] = df["lat_grid"].round(4)
df["lon_grid"] = df["lon_grid"].round(4)

# ---------------- CLEAN DATA ----------------
# Drop rows with missing critical values
df = df.dropna(subset=["Afla", "Fum", "lat", "lon"])

# ---------------- HELPER FUNCTION ----------------
def mode_func(x):
    return x.mode().iloc[0] if not x.mode().empty else np.nan

# ---------------- DEFINE FEATURES ----------------
categorical_cols = [
    "Harvestdate",
    "Sowdate",

]

# keep only columns that exist
categorical_cols = [c for c in categorical_cols if c in df.columns]

# ---------------- GROUPING ----------------
group_cols = ["lat_grid", "lon_grid", "Country", "Crop", "Color", "Tillage", "Fertilizer", "Awareness", "Sowmethod", "Prevtime", "Prevcrop", "Biocide"]

agg_dict = {
    "lat": "count",
    "Afla": "mean",
    "Fum": "mean",
    "Ota": "mean",
    "Zen": "mean",
    "Don": "mean",
    "Cultdays": "mean",
    #add soil features to agg dict
    **{col: "mean" for col in soil_items}
}

# add categorical aggregation
for col in categorical_cols:
    agg_dict[col] = mode_func

grid_df = df.groupby(group_cols).agg(agg_dict).reset_index()

# rename count column
grid_df = grid_df.rename(columns={"lat": "n_points"})

# ---------------- GRID CENTER ----------------
grid_df["Latitude"] = grid_df["lat_grid"] + latgrid / 2
grid_df["Longitude"] = grid_df["lon_grid"] + longgrid / 2

#

# ---------------- SAVE ----------------
grid_df.to_csv("data/processed/grid_data.csv", index=False)

print("Grid aggregation complete")
print(grid_df.head())
print("Shape:", grid_df.shape)
# the supposed json file loo like: ["Ota", "Zen", "Don", "Cultdays", "harvest_month", "sow_month", "Color_multicolor", "Color_red", "Color_white", "Color_yellow", "Tillage_notillage", "Tillage_shallowtillage", "Biocide_no", "Biocide_pesticide", "Fertilizer_notapplied", "Seedprep_notapplied", "Awareness_notaware", "Sowmethod_disperse", "Prevtime_lastseason", "Prevtime_lastyear", "Prevcrop_cottonflower", "Prevcrop_legume", "Prevcrop_mix", "Prevcrop_tuber"]

