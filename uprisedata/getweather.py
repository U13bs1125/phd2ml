import pandas as pd
import requests
import time
from tqdm import tqdm
import numpy as np

df = pd.read_csv("data/preprocessed/2024pg.csv")
# ✅ Updated NASA POWER parameters
parameters = [
    "T2M",                  # Temperature at 2m
    "PRECTOTCORR",          # Precipitation
    "RH2M",                 # Relative humidity
    "ALLSKY_SFC_SW_DWN",    # Solar radiation
    "GWETROOT",             # Root zone soil wetness
    "EVPTRNS",              # Evapotranspiration
    "WS2M"                  # Wind speed at 2m
]

def fetch_nasa_power_weekly(lat, lon, sow, harvest):

    if pd.isna(lat) or pd.isna(lon) or pd.isna(sow) or pd.isna(harvest):
        return None

    # ✅ Start 1 week BEFORE sowing
    start = sow - pd.Timedelta(days=7)
    end = harvest

    start_str = start.strftime('%Y%m%d')
    end_str = end.strftime('%Y%m%d')

    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters={','.join(parameters)}"
        f"&community=AG"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={start_str}"
        f"&end={end_str}"
        f"&format=JSON"
    )

    try:
        r = requests.get(url, timeout=60)
        data = r.json()

        params = data["properties"]["parameter"]

        daily = pd.DataFrame(params)
        daily.index = pd.to_datetime(daily.index, format="%Y%m%d")
        daily = daily.sort_index()

        row_out = {}

        # ===============================
        # ✅ WEEKLY AGGREGATION
        # ===============================
        total_days = (end - start).days + 1
        n_weeks = int(np.ceil(total_days / 7))

        for w in range(n_weeks):

            week_start = start + pd.Timedelta(days=w * 7)
            week_end = week_start + pd.Timedelta(days=6)

            week_data = daily.loc[week_start:week_end]

            if len(week_data) == 0:
                continue

            # naming: w0 = week before sowing
            week_label = f"w{w}"

            for p in parameters:
                row_out[f"{p}_{week_label}"] = week_data[p].mean()

        return row_out

    except Exception as e:
        print("Error:", e)
        return None


# ===================================
# LOOP THROUGH OBSERVATIONS
# ===================================

weather_rows = []

for idx, row in tqdm(df.iterrows(), total=len(df)):

    lat = row["Latitude"]
    lon = row["Longitude"]
    sow = pd.to_datetime(row["Sowdate"])
    har = pd.to_datetime(row["Harvestdate"])

    data = fetch_nasa_power_weekly(lat, lon, sow, har)

    if data:
        data["Latitude"] = lat
        data["Longitude"] = lon
        data["Sowdate"] = sow
        data["Harvestdate"] = har

        weather_rows.append(data)

    time.sleep(1)  # API safety

weather_df = pd.DataFrame(weather_rows)

weather_df.to_csv("data/preprocessed/2024pw_weekly.csv", index=False)

print("✅ Weekly weather features generated.")