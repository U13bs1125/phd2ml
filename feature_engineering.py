import pandas as pd
import numpy as np
import re

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("data/preprocessed/2024pw_daily8p120.csv")

# ===============================
# HELPERS
# ===============================
def get_days(prefix):
    pattern = re.compile(f"{prefix}_D(-?\d+)")
    return sorted([
        int(pattern.search(c).group(1))
        for c in df.columns if pattern.search(c)
    ])

days = get_days("T2M_MAX")

# ===============================
# EXTRACT TIME SERIES
# ===============================
def extract_series(var):
    return np.array([
        df[f"{var}_D{d}"].values if f"{var}_D{d}" in df.columns else np.zeros(len(df))
        for d in days
    ])

Tmax = extract_series("T2M_MAX")
Tmin = extract_series("T2M_MIN")
Rain = extract_series("PRECTOTCORR")
RH = extract_series("RH2M")
Rad = extract_series("ALLSKY_SFC_SW_DWN")
SM = extract_series("GWETROOT")
ET = extract_series("EVPTRNS")
WS = extract_series("WS2M")

Tmean = (Tmax + Tmin) / 2

# ===============================
# FEATURE ENGINEERING
# ===============================
feat = pd.DataFrame(index=df.index)

# ===============================
# 🌡 TEMPERATURE FEATURES
# ===============================
feat["temp_mean"] = Tmean.mean(axis=0)
feat["temp_std"] = Tmean.std(axis=0)
feat["temp_max"] = Tmax.max(axis=0)
feat["temp_min"] = Tmin.min(axis=0)

feat["heat_days"] = (Tmax > 35).sum(axis=0)
feat["cold_days"] = (Tmin < 10).sum(axis=0)
feat["warm_nights"] = (Tmin > 20).sum(axis=0)

# ===============================
# 💧 RAIN + DROUGHT
# ===============================
feat["rain_total"] = Rain.sum(axis=0)
feat["rain_mean"] = Rain.mean(axis=0)
feat["rain_std"] = Rain.std(axis=0)

feat["dry_days"] = (Rain < 1).sum(axis=0)

# longest dry spell
def max_consecutive_dry(r):
    return max((len(list(g)) for k, g in groupby(r < 1) if k), default=0)

from itertools import groupby
feat["max_dry_spell"] = [
    max_consecutive_dry(Rain[:, i]) for i in range(Rain.shape[1])
]

# ===============================
# 🌫 HUMIDITY
# ===============================
feat["rh_mean"] = RH.mean(axis=0)
feat["rh_high_days"] = (RH > 80).sum(axis=0)
feat["rh_low_days"] = (RH < 50).sum(axis=0)

# CRITICAL: fungal favorable condition
feat["rh_temp_interaction"] = np.mean((RH > 80) & (Tmean > 25), axis=0)

# ===============================
# 🌞 RADIATION
# ===============================
feat["rad_mean"] = Rad.mean(axis=0)
feat["rad_high_days"] = (Rad > 250).sum(axis=0)

# ===============================
# 🌱 SOIL + WATER
# ===============================
feat["soil_moisture_mean"] = SM.mean(axis=0)
feat["soil_moisture_low_days"] = (SM < 0.3).sum(axis=0)

feat["et_mean"] = ET.mean(axis=0)
feat["et_high_days"] = (ET > 5).sum(axis=0)

# moisture stress index
feat["drought_index"] = (ET.mean(axis=0) / (SM.mean(axis=0) + 1e-5))

# ===============================
# 🌬 WIND (DISPERSAL)
# ===============================
feat["wind_mean"] = WS.mean(axis=0)
feat["wind_high_days"] = (WS > 3).sum(axis=0)

# dispersal proxy
feat["dry_windy_days"] = ((Rain < 1) & (WS > 2)).sum(axis=0)

# ===============================
# 🌱 GROWTH STAGE FEATURES
# ===============================
def stage_mean(arr, start, end):
    idx = [i for i, d in enumerate(days) if start <= d <= end]
    if len(idx) == 0:
        return np.zeros(arr.shape[1])
    return arr[idx].mean(axis=0)

# EARLY (0–30)
feat["temp_early"] = stage_mean(Tmean, 0, 30)
feat["rain_early"] = stage_mean(Rain, 0, 30)
feat["rh_early"] = stage_mean(RH, 0, 30)

# MID (30–80)
feat["temp_mid"] = stage_mean(Tmean, 30, 80)
feat["rain_mid"] = stage_mean(Rain, 30, 80)
feat["rh_mid"] = stage_mean(RH, 30, 80)

# LATE (80–120)
feat["temp_late"] = stage_mean(Tmean, 80, 120)
feat["rain_late"] = stage_mean(Rain, 80, 120)
feat["rh_late"] = stage_mean(RH, 80, 120)

# ===============================
# 🌾 TOXIN-SPECIFIC RISK FEATURES
# ===============================

# aflatoxin favors HOT + DRY
feat["afla_risk_index"] = np.mean(
    (Tmean > 30) & (RH < 70),
    axis=0
)

# fumonisin favors WET + HUMID
feat["fum_risk_index"] = np.mean(
    (RH > 80) & (Rain > 2),
    axis=0
)

# ===============================
# SAVE
# ===============================
df_final = feat.copy()

df_final.to_csv("data/processed/engineered_features.csv", index=False)

print("✅ Feature engineering complete")
print("New features:", feat.columns.tolist())