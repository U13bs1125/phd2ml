import pandas as pd
import numpy as np
import re

# ============================================
# STEP 1: FIX COLUMN NAMING (D3 → D-3)
# ============================================
def rename_days(col):
    match = re.search(r'D(\d+)$', col)
    if match:
        d = int(match.group(1))
        return col.replace(f"D{d}", f"D-{d}")
    return col


# ============================================
# STEP 2: EXTRACT DAY INDEX
# ============================================
def get_day_index(col):
    match = re.search(r'D(-?\d+)', col)
    return int(match.group(1)) if match else None


# ============================================
# STEP 3: GDD FUNCTION
# ============================================
def get_gdd(T, base=6, cut=40):
    T_adj = np.clip(T, base, cut)
    return np.maximum(0, T_adj - base)


# ============================================
# STEP 4: MAX DRY SPELL
# ============================================
def max_consecutive_dry_days(arr):
    max_len = 0
    cur = 0
    for val in arr:
        if val == 0:
            cur += 1
            max_len = max(max_len, cur)
        else:
            cur = 0
    return max_len


# ============================================
# STEP 5: MAIN FEATURE ENGINEERING
# ============================================
def engineer_features(df):

    df = df.copy()

    # ----------------------------------------
    # IDENTIFY & SORT DAYS
    # ----------------------------------------
    all_days = sorted(
        set([get_day_index(c) for c in df.columns if "_D" in c])
    )
    all_days = [d for d in all_days if d is not None]

    print(f"Detected days: {all_days[:5]} ... {all_days[-5:]}")

    # ----------------------------------------
    # BUILD TIME SERIES
    # ----------------------------------------
    TMAX = np.array([df[f"T2M_MAX_D{d}"] for d in all_days]).T
    TMIN = np.array([df[f"T2M_MIN_D{d}"] for d in all_days]).T
    R = np.array([df[f"PRECTOTCORR_D{d}"] for d in all_days]).T
    RH = np.array([df[f"RH2M_D{d}"] for d in all_days]).T
    RAD = np.array([df[f"ALLSKY_SFC_SW_DWN_D{d}"] for d in all_days]).T
    SM = np.array([df[f"GWETROOT_D{d}"] for d in all_days]).T
    ET = np.array([df[f"EVPTRNS_D{d}"] for d in all_days]).T
    WS = np.array([df[f"WS2M_D{d}"] for d in all_days]).T

    # Average temperature
    T = (TMAX + TMIN) / 2

    # ----------------------------------------
    # SPLIT PRE / POST SOWING
    # ----------------------------------------
    pre_idx = [i for i, d in enumerate(all_days) if d < 0]
    post_idx = [i for i, d in enumerate(all_days) if d >= 0]

    # ============================================
    # TEMPERATURE FEATURES
    # ============================================
    df["T_mean"] = np.nanmean(T, axis=1)
    df["T_max_season"] = np.nanmax(T, axis=1)
    df["T_min_season"] = np.nanmin(T, axis=1)
    df["T_var"] = np.nanstd(T, axis=1)

    df["heat_days"] = (T > 32).sum(axis=1)
    df["extreme_heat_days"] = (T > 35).sum(axis=1)
    df["optimal_temp_days"] = ((T > 25) & (T < 35)).sum(axis=1)

    GDD = get_gdd(T)
    df["GDD_total"] = GDD.sum(axis=1)
    df["GDD_early"] = GDD[:, post_idx][:, :20].sum(axis=1)
    df["GDD_mid"] = GDD[:, post_idx][:, 20:60].sum(axis=1)
    df["GDD_late"] = GDD[:, post_idx][:, -20:].sum(axis=1)

    # ============================================
    # RAINFALL / DROUGHT
    # ============================================
    df["rain_total"] = np.nansum(R[:, post_idx], axis=1)
    df["rain_pre_sow"] = np.nansum(R[:, pre_idx], axis=1)

    df["rain_std"] = np.nanstd(R, axis=1)
    df["dry_days"] = (R[:, post_idx] == 0).sum(axis=1)

    df["max_dry_spell"] = np.apply_along_axis(
        max_consecutive_dry_days, 1, R[:, post_idx]
    )

    df["late_drought"] = (R[:, post_idx][:, -20:] == 0).sum(axis=1)
    df["early_drought"] = (R[:, post_idx][:, :20] == 0).sum(axis=1)

    # ============================================
    # HUMIDITY
    # ============================================
    df["RH_mean"] = np.nanmean(RH[:, post_idx], axis=1)
    df["RH_low_days"] = (RH[:, post_idx] < 60).sum(axis=1)
    df["RH_high_days"] = (RH[:, post_idx] > 80).sum(axis=1)

    # ============================================
    # RADIATION
    # ============================================
    df["RAD_total"] = np.nansum(RAD[:, post_idx], axis=1)
    df["RAD_mean"] = np.nanmean(RAD[:, post_idx], axis=1)
    df["high_rad_days"] = (
        RAD[:, post_idx] > np.nanmean(RAD[:, post_idx], axis=1, keepdims=True)
    ).sum(axis=1)

    # ============================================
    # SOIL MOISTURE / WATER STRESS
    # ============================================
    df["soil_moisture_mean"] = np.nanmean(SM[:, post_idx], axis=1)
    df["soil_moisture_pre"] = np.nanmean(SM[:, pre_idx], axis=1)

    df["soil_moisture_low_days"] = (SM[:, post_idx] < 0.3).sum(axis=1)

    df["ET_total"] = np.nansum(ET[:, post_idx], axis=1)
    df["water_deficit"] = df["ET_total"] / (df["rain_total"] + 1)

    # ============================================
    # WIND
    # ============================================
    df["wind_mean"] = np.nanmean(WS[:, post_idx], axis=1)
    df["wind_high_days"] = (WS[:, post_idx] > 3).sum(axis=1)

    # ============================================
    # INTERACTIONS
    # ============================================
    df["temp_rh_interaction"] = np.nanmean(T * RH, axis=1)
    df["temp_drought"] = np.nanmean(T * (R == 0), axis=1)
    df["rad_drought"] = np.nanmean(RAD * (R == 0), axis=1)

    df["stress_index"] = df["heat_days"] * df["dry_days"]

    # ============================================
    # STAGE-SPECIFIC (VERY IMPORTANT)
    # ============================================
    df["flowering_drought"] = (R[:, post_idx][:, 30:60] == 0).sum(axis=1)
    df["grain_filling_heat"] = (T[:, post_idx][:, 60:100] > 32).sum(axis=1)

    # ============================================
    # SOIL STATIC FEATURES
    # ============================================
    df["soil_texture"] = df["sand_content"] / (df["clay_content"] + 1)
    df["organic_ratio"] = df["carbon_organic"] / (df["carbon_total"] + 1)

    df["cec_moisture"] = df["cation_exchange_capacity"] * df["clay_content"]
    df["bulk_density_stress"] = df["bulk_density"] * df["sand_content"]

    # ============================================
    # AGRONOMY
    # ============================================
    df["crop_duration"] = df["Cultdays"]
    df["late_planting"] = (df["sow_month"] > 6).astype(int)

    df["fertilizer_used"] = (df["Fertilizer_notapplied"] == 0).astype(int)
    df["legume_prev"] = df["Prevcrop_legume"]
    df["tillage_intensity"] = df["Tillage_shallowtillage"]

    return df


# ============================================
# RUN SCRIPT
# ============================================
if __name__ == "__main__":

    df = pd.read_csv("data/preprocessed/2024pw_daily8p120.csv")

    # Fix column names
    df.columns = [rename_days(c) for c in df.columns]

    # Engineer features
    df_new = engineer_features(df)

    # Save
    df_new.to_csv("data/processed/engineered_features.csv", index=False)

    print("✅ Feature engineering completed")
    print(df_new.shape)