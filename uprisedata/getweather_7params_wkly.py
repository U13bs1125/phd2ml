import pandas as pd
import numpy as np
import requests
import time
from tqdm import tqdm

# =========================
# CONFIG
# =========================
MAX_WEEKS = 20

parameters = [
    "T2M","PRECTOTCORR","RH2M",
    "ALLSKY_SFC_SW_DWN","GWETROOT","EVPTRNS","WS2M"
]

# =========================
# FUNCTIONS
# =========================
def get_gdd(T, Tbase=6, Tcut=30):
    if pd.isna(T): return 0
    return max(0, np.clip(T, Tbase, Tcut) - Tbase)

def afla_response(temp):
    A,B,C=4.84,1.32,5.59
    Teq=(temp-10)/(47-10)
    return np.nan_to_num((A*(Teq**B)*(1-Teq))**C)

def growth_response(temp):
    A,B,C=5.98,1.70,1.43
    Teq=(temp-5)/(48-5)
    return np.nan_to_num((A*(Teq**B)*(1-Teq))**C)

def compute_ari(T,R,RH):
    DIS = 1 if (R < 1 and RH < 80) else 0
    return afla_response(T)*growth_response(T)*DIS

# =========================
# NASA FETCH
# =========================
def fetch_features(lat, lon, sow, harvest):

    start = sow - pd.Timedelta(days=7)
    end = harvest

    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters={','.join(parameters)}&community=AG&longitude={lon}&latitude={lat}&start={start.strftime('%Y%m%d')}&end={end.strftime('%Y%m%d')}&format=JSON"

    try:
        r = requests.get(url, timeout=60)
        data = r.json()["properties"]["parameter"]

        daily = pd.DataFrame(data)
        daily.index = pd.to_datetime(daily.index, format="%Y%m%d")

        row_out = {}

        # ================= WEEKLY =================
        for w in range(MAX_WEEKS):

            week_start = start + pd.Timedelta(days=w * 7)
            week_end = week_start + pd.Timedelta(days=6)

            week_data = daily.loc[week_start:week_end]

            if len(week_data) == 0:
                for p in parameters:
                    row_out[f"{p}_w{w}"] = 0
                row_out[f"ARI_w{w}"] = 0
                continue

            for p in parameters:
                row_out[f"{p}_w{w}"] = week_data[p].mean()

            T = week_data["T2M"].mean()
            R = week_data["PRECTOTCORR"].mean()
            RH = week_data["RH2M"].mean()

            row_out[f"ARI_w{w}"] = compute_ari(T, R, RH)

        # ================= POST-SOWING ARI =================
        grow = daily.loc[sow:harvest]

        ari_post = np.sum([
            compute_ari(t, r, rh)
            for t, r, rh in zip(
                grow["T2M"], grow["PRECTOTCORR"], grow["RH2M"]
            )
        ])

        row_out["ARI_post"] = ari_post

        return row_out

    except Exception as e:
        print("Error:", e)
        return None


# =========================
# BUILD DATASET
# =========================
if __name__ == "__main__":

    df = pd.read_csv("data/processed/grid_data.csv")

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        data = fetch_features(
            row["Latitude"],
            row["Longitude"],
            pd.to_datetime(row["Sowdate"]),
            pd.to_datetime(row["Harvestdate"])
        )

        if data is not None:
            data["Aflac"] = int(row["Afla"] > 4)
            data['Fumc'] = int(row["Fum"] > 4000)
            rows.append(data)

        time.sleep(1)

    final_df = pd.DataFrame(rows)

    final_df.to_csv("data/processed/7params_weekly.csv", index=False)

    print(" Dataset saved to data/processed/7params_weekly.csv")
    print(final_df.shape)