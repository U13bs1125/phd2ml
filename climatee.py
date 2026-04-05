import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# ---------------- LOAD ----------------
def load_data():
    df = pd.read_csv("data/preprocessed/2024pg.csv")

    df["Aflac"] = (df["Afla"] > 10).astype(int)
    df["Fumc"] = (df["Fum"] > 4000).astype(int)

    return df


# ---------------- CLIMATE SCENARIOS ----------------
def apply_climate_scenario(df, scenario):

    df = df.copy()

    t2m_cols = [c for c in df.columns if "T2M" in c]
    prec_cols = [c for c in df.columns if "PRECTOT" in c]

    if scenario == "ssp245":
        for col in t2m_cols:
            df[col] *= 1.03
        for col in prec_cols:
            df[col] *= 1.05

    elif scenario == "ssp585":
        for col in t2m_cols:
            df[col] *= 1.08
        for col in prec_cols:
            df[col] *= 1.10

    return df


# ---------------- MODEL ----------------
def load_model(name):
    return joblib.load(f"artifacts/models/{name}.pkl")


def predict(df, model):
    features = model.named_steps["scaler"].feature_names_in_

    X = df.select_dtypes(include=np.number).copy()

    for col in features:
        if col not in X.columns:
            X[col] = 0

    X = X[features]

    return model.predict_proba(X)[:, 1]


# ---------------- BUILD DATASET ----------------
def build_dataset(df):

    models = {
        "Aflac": load_model("allafrf"),
        "Fumc": load_model("allfurf")
    }

    results = []

    for scenario in ["baseline", "ssp245", "ssp585"]:

        df_scn = apply_climate_scenario(df, scenario)

        for target in ["Aflac", "Fumc"]:

            probs = predict(df_scn, models[target])

            df_tmp = df_scn.copy()
            df_tmp["risk"] = probs

            # REGION LEVEL aggregation
            grouped = df_tmp.groupby(["Region", "Country"]).agg(
                risk_mean=("risk", "mean"),
                risk_std=("risk", "std")
            ).reset_index()

            grouped["Scenario"] = scenario
            grouped["Target"] = target

            results.append(grouped)

    return pd.concat(results, ignore_index=True)


# ---------------- RELATIVE CHANGE ----------------
def compute_changes(df):

    base = df[df["Scenario"] == "baseline"][["Region", "Country", "Target", "risk_mean"]]
    base = base.rename(columns={"risk_mean": "risk_base"})

    df = df.merge(base, on=["Region", "Country", "Target"], how="left")

    df["risk_change"] = (df["risk_mean"] - df["risk_base"]) / df["risk_base"] * 100
    df.loc[df["Scenario"] == "baseline", "risk_change"] = 0

    return df


# ---------------- HOTSPOTS ----------------
def add_hotspots(df):
    df["hotspot"] = (df["risk_change"] > 20).astype(int)
    return df


# ---------------- SCENARIO DIFFERENCE ----------------
def compute_difference(df):

    ssp245 = df[df["Scenario"] == "ssp245"]
    ssp585 = df[df["Scenario"] == "ssp585"]

    merged = ssp585.merge(
        ssp245,
        on=["Region", "Country", "Target"],
        suffixes=("_585", "_245")
    )

    merged["scenario_diff"] = merged["risk_mean_585"] - merged["risk_mean_245"]

    return merged


# ---------------- PLOT ----------------
def plot_maps(df):

    df = compute_changes(df)
    df = add_hotspots(df)
    diff_df = compute_difference(df)

    scenarios = ["baseline", "ssp245", "ssp585"]
    targets = ["Aflac", "Fumc"]

    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=[
            f"{t} - {s}" for t in targets for s in scenarios
        ] + [
            "Hotspots (SSP585)",
            "Uncertainty (Std Dev)",
            "Scenario Difference (585-245)"
        ],
        specs=[
            [{"type": "choropleth"}]*3,
            [{"type": "choropleth"}]*3,
            [{"type": "choropleth"}]*3
        ]
    )

    # ---- MAIN MAPS ----
    for i, target in enumerate(targets):
        for j, scenario in enumerate(scenarios):

            data = df[(df["Target"] == target) & (df["Scenario"] == scenario)]

            z = data["risk_mean"] if scenario == "baseline" else data["risk_change"]

            fig.add_trace(
                go.Choropleth(
                    locations=data["Country"],
                    locationmode="country names",
                    z=z,
                    colorscale="Viridis" if scenario=="baseline" else "RdYlGn_r",
                    zmin=0 if scenario=="baseline" else -50,
                    zmax=1 if scenario=="baseline" else 100,
                    showscale=False
                ),
                row=i+1,
                col=j+1
            )

    # ---- HOTSPOTS ----
    hotspot = df[(df["Scenario"] == "ssp585") & (df["Target"] == "Aflac")]

    fig.add_trace(
        go.Choropleth(
            locations=hotspot["Country"],
            locationmode="country names",
            z=hotspot["hotspot"],
            colorscale="Reds",
            zmin=0, zmax=1,
            showscale=False
        ),
        row=3, col=1
    )

    # ---- UNCERTAINTY ----
    unc = df[(df["Scenario"] == "ssp585") & (df["Target"] == "Aflac")]

    fig.add_trace(
        go.Choropleth(
            locations=unc["Country"],
            locationmode="country names",
            z=unc["risk_std"],
            colorscale="Blues",
            showscale=False
        ),
        row=3, col=2
    )

    # ---- SCENARIO DIFFERENCE ----
    diff = diff_df[diff_df["Target"] == "Aflac"]

    fig.add_trace(
        go.Choropleth(
            locations=diff["Country"],
            locationmode="country names",
            z=diff["scenario_diff"],
            colorscale="RdBu",
            zmid=0,
            showscale=True
        ),
        row=3, col=3
    )

    fig.update_layout(
        title="Climate Change Impact on Mycotoxin Risk (Regional Analysis)",
        height=1000
    )

    for r in range(1, 4):
        for c in range(1, 4):
            fig.update_geos(
                scope="africa",
                showframe=False,
                showcoastlines=False,
                row=r,
                col=c
            )

    fig.show()


# ---------------- MAIN ----------------
if __name__ == "__main__":

    df = load_data()
    df_map = build_dataset(df)
    plot_maps(df_map)
