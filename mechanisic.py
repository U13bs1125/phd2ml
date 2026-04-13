import numpy as np
import pandas as pd


# =====================================
# GDD FUNCTION (from original logic)
# =====================================
def get_gdd(T, Tbase=6, Tcut=30):
    if pd.isna(T):
        return 0
    T_adj = np.clip(T, Tbase, Tcut)
    return max(0, T_adj - Tbase)


# =====================================
# BIOLOGICAL FUNCTIONS
# =====================================
def afla_response(temp):
    A, B, C = 4.84, 1.32, 5.59
    T_max, T_min = 47, 10
    Teq = (temp - T_min) / (T_max - T_min)
    return np.nan_to_num((A * (Teq ** B) * (1 - Teq)) ** C)


def growth_response(temp):
    A, B, C = 5.98, 1.70, 1.43
    T_max, T_min = 48, 5
    Teq = (temp - T_min) / (T_max - T_min)
    return np.nan_to_num((A * (Teq ** B) * (1 - Teq)) ** C)


def dispersal(rain, rh):
    dis_rain = 1 if rain == 0 else 0
    dis_rh = 1 if rh < 80 else 0
    return 1 if (dis_rain + dis_rh == 2) else 0


# =====================================
# MAIN FUNCTION
# =====================================
def compute_ari_from_features(df, n_bins=5):

    df = df.copy()

    # Get ordered stages (VERY IMPORTANT)
    stage_cols = sorted([col for col in df.columns if col.startswith("T2M_")])

    stage_names = [col.replace("T2M_", "") for col in stage_cols]

    results = []

    for _, row in df.iterrows():

        # ---------------------------------
        # Reconstruct time series
        # ---------------------------------
        T_series = []
        R_series = []
        RH_series = []

        for stage in stage_names:
            T_series.append(row[f"T2M_{stage}"])
            R_series.append(row.get(f"PRECTOTCORR_{stage}", 0))
            RH_series.append(row.get(f"RH2M_{stage}", 0))

        T_series = np.array(T_series, dtype=float)
        R_series = np.array(R_series, dtype=float)
        RH_series = np.array(RH_series, dtype=float)

        # ---------------------------------
        # Compute GDD (daily)
        # ---------------------------------
        GDD_daily = np.array([get_gdd(T) for T in T_series])

        # Add emergence correction
        if len(GDD_daily) > 0:
            GDD_daily[0] += 50

        GDD_cum = np.cumsum(GDD_daily)

        # ---------------------------------
        # Define GDD bins (like AFLA-maize)
        # ---------------------------------
        temp_bins = np.linspace(750, 1500, n_bins + 1)

        ari_bins = []

        # ---------------------------------
        # Loop through bins
        # ---------------------------------
        for j in range(len(temp_bins) - 1):

            idx = np.where(
                (GDD_cum > temp_bins[j]) &
                (GDD_cum < temp_bins[j + 1])
            )[0]

            if len(idx) == 0:
                ari_bins.append(0)
                continue

            Temp = T_series[idx]
            Rain = R_series[idx]
            RH = RH_series[idx]

            # ---------------------------------
            # DISPERSAL
            # ---------------------------------
            DIS_rain = np.where(Rain == 0, 1, 0)
            DIS_RH = np.where(RH < 80, 1, 0)
            DIS = DIS_rain + DIS_RH
            DIS = np.where(DIS == 2, 1, 0)

            # ---------------------------------
            # AFLA + GROWTH
            # ---------------------------------
            AFLA = afla_response(Temp)
            GROWTH = growth_response(Temp)

            risk = AFLA * GROWTH * DIS

            ari_bins.append(np.sum(risk))

        ari_total = np.sum(ari_bins)

        results.append(ari_bins + [ari_total])

    # ---------------------------------
    # Output DataFrame
    # ---------------------------------
    columns = [f"ARI_bin{i+1}" for i in range(n_bins)] + ["ARI_total"]

    ari_df = pd.DataFrame(results, columns=columns).astype(float)

    ari_df.to_csv("data/ari_values.csv", index=False)

    df_out = pd.concat([df.reset_index(drop=True), ari_df], axis=1)

    return df_out, ari_df


# =====================================
# RUN SCRIPT
# =====================================
if __name__ == "__main__":
    df = pd.read_csv("data/processed/2024pg.csv")

    df_with_ari, ari_df = compute_ari_from_features(df)

    df_with_ari.to_csv("data/weather_soil_agro_with_ari.csv", index=False)

    print(" ARI bins + total computed and saved.")