import pandas as pd
import requests
import time
from tqdm import tqdm
import numpy as np

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/preprocessed/2024pg.csv")

# =========================
# NASA POWER PARAMETERS
# =========================
parameters = [
    "T2M_MAX",              # ✅ Max temperature
    "T2M_MIN",              # ✅ Min temperature
    "PRECTOTCORR",          # Precipitation
    "RH2M",                 # Relative humidity
    "ALLSKY_SFC_SW_DWN",    # Solar radiation
    "GWETROOT",             # Root zone soil wetness
    "EVPTRNS",              # Evapotranspiration
    "WS2M"                  # Wind speed
]

# =========================
# FETCH DAILY DATA
# =========================
def fetch_nasa_power_daily(lat, lon, sow):

    if pd.isna(lat) or pd.isna(lon) or pd.isna(sow):
        return None

    # ✅ from 3 days BEFORE sowing to 120 days AFTER
    start = sow - pd.Timedelta(days=3)
    end = sow + pd.Timedelta(days=120)

    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters={','.join(parameters)}"
        f"&community=AG"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={start.strftime('%Y%m%d')}"
        f"&end={end.strftime('%Y%m%d')}"
        f"&format=JSON"
    )

    try:
        r = requests.get(url)
        data = r.json()

        params = data["properties"]["parameter"]

        daily = pd.DataFrame(params)
        daily.index = pd.to_datetime(daily.index, format="%Y%m%d")
        daily = daily.sort_index()

        row_out = {}

        # =========================
        # CREATE RELATIVE DAY FEATURES
        # =========================
        for offset in range(-3, 121):   # D-3 to D120

            current_day = sow + pd.Timedelta(days=offset)

            if current_day in daily.index:
                day_data = daily.loc[current_day]

                for p in parameters:
                    value = day_data.get(p, np.nan)
                    row_out[f"{p}_D{offset}"] = value

            else:
                # if missing day → fill NaN
                for p in parameters:
                    row_out[f"{p}_D{offset}"] = np.nan

        return row_out

    except Exception as e:
        print("Error:", e)
        return None


# =========================
# LOOP THROUGH DATA
# =========================
weather_rows = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    lat = row["Latitude"]
    lon = row["Longitude"]
    sow = pd.to_datetime(row["Sowdate"])

    data = fetch_nasa_power_daily(lat, lon, sow)

    if data:
        data["Latitude"] = lat
        data["Longitude"] = lon
        data["Sowdate"] = sow

        weather_rows.append(data)

    time.sleep(1)  # API safety

# =========================
# SAVE OUTPUT
# =========================
weather_df = pd.DataFrame(weather_rows)

weather_df.to_csv("data/preprocessed/2024pw_daily8p120.csv", index=False)

print("Daily aligned weather features generated.")
print(weather_df.shape)