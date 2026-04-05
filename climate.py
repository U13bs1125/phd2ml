# ================================
# INSTALL (run once)
# ================================
# pip install -U kaleido plotly

import pandas as pd
import numpy as np
import joblib
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================================
# LOAD DATA
# ================================
def load_data():
    df = pd.read_csv("data/preprocessed/2024pg.csv")

    df["Aflac"] = (df["Afla"] > 10).astype(int)
    df["Fumc"] = (df["Fum"] > 4000).astype(int)

    # Clean country names (VERY IMPORTANT)
    country_map = {
        'ivory coast': "Cote d'Ivoire",
        'drc': 'Democratic Republic of the Congo',
        'cameroun': 'Cameroon'
    }

    df["Country"] = df["Country"].str.strip().str.title()
    df["Country"] = df["Country"].replace(country_map)

    return df

# ================================
# MODEL LOADING
# ================================
def load_model(model_id):
    path = f"artifacts/models/{model_id}.pkl"
    print(f"Loaded → {model_id}")
    return joblib.load(path)

# ================================
# FEATURE ALIGNMENT
# ================================
def prepare_X(df, features):
    X = df.select_dtypes(include=np.number).fillna(0)

    for col in features:
        if col not in X.columns:
            X[col] = 0

    return X[features]

# ================================
# CLIMATE SCENARIOS
# ================================
def apply_climate(df, scenario, year):

    df = df.copy()

    if scenario == "baseline":
        return df

    if scenario == "ssp245":
        temp = 1.8 if year == 2050 else 1.0
        rain = 1.05

    elif scenario == "ssp585":
        temp = 3.0 if year == 2050 else 1.5
        rain = 1.10

    for col in df.columns:
        if "T2M" in col:
            df[col] += temp
        if "PRECTOT" in col:
            df[col] *= rain

    return df

# ================================
# PREDICT
# ================================
def predict(df, model, target):

    features = model.named_steps["scaler"].feature_names_in_

    X = prepare_X(df, features)
    probs = model.predict_proba(X)[:, 1]

    return probs

# ================================
# BUILD DATA
# ================================
def build_dataset(df):

    models = {
        "Aflac": load_model("allafrf"),
        "Fumc": load_model("allfurf")
    }

    scenarios = [
        ("baseline", 2024),
        ("ssp245", 2050),
        ("ssp585", 2050)
    ]

    results = []

    for target, model in models.items():

        base_probs = predict(df, model, target)

        for scen, year in scenarios:

            df_future = apply_climate(df, scen, year)
            probs = predict(df_future, model, target)

            change = (probs - base_probs) / (base_probs + 1e-6) * 100

            temp = df[["Country"]].copy()
            temp["Target"] = target
            temp["Scenario"] = scen
            temp["Year"] = year
            temp["Risk"] = probs
            temp["RiskChange"] = change

            results.append(temp)

    return pd.concat(results, ignore_index=True)

# ================================
# AGGREGATE COUNTRY LEVEL
# ================================
def aggregate_country(df):

    return df.groupby(
        ["Country", "Target", "Scenario", "Year"]
    ).agg(
        Risk=("Risk", "mean"),
        RiskChange=("RiskChange", "mean")
    ).reset_index()

# ================================
# PLOT MAPS (FIXED)
# ================================
def plot_maps(df):

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Aflac Baseline",
            "Aflac SSP2-4.5 (2050)",
            "Aflac SSP5-8.5 (2050)",
            "Fumc Baseline",
            "Fumc SSP2-4.5 (2050)",
            "Fumc SSP5-8.5 (2050)"
        ],
        specs=[[{"type": "choropleth"}]*3,
               [{"type": "choropleth"}]*3]
    )

    scenarios = [
        ("baseline", 2024),
        ("ssp245", 2050),
        ("ssp585", 2050)
    ]

    targets = ["Aflac", "Fumc"]

    for r, target in enumerate(targets, start=1):
        for c, (scen, year) in enumerate(scenarios, start=1):

            subset = df[
                (df["Target"] == target) &
                (df["Scenario"] == scen) &
                (df["Year"] == year)
            ]

            fig.add_trace(
                go.Choropleth(
                    locations=subset["Country"],
                    locationmode="country names",
                    z=subset["RiskChange"],
                    colorscale="RdYlGn_r",
                    zmin=-100,
                    zmax=100,
                    coloraxis="coloraxis",
                    showscale=(r == 1 and c == 3)
                ),
                row=r, col=c
            )

    fig.update_layout(
        title="Climate Change Impact on Mycotoxin Risk (% Change)",
        height=700,
        coloraxis=dict(colorbar=dict(title="% Risk Change")),
        margin=dict(l=10, r=10, t=60, b=10)
    )

    # Apply Africa scope
    for i in range(6):
        fig.update_geos(
            scope="africa",
            showcoastlines=False,
            showframe=False,
            row=(i // 3) + 1,
            col=(i % 3) + 1
        )

    os.makedirs("results/plots", exist_ok=True)

    fig.show()
    fig.write_image("results/plots/climate_maps.png", scale=2)

# ================================
# MAIN
# ================================
if __name__ == "__main__":

    df = load_data()

    df_map = build_dataset(df)

    df_country = aggregate_country(df_map)

    plot_maps(df_country)