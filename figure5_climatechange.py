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

    # 🔥 Automatically detect climate columns
    t2m_cols = [c for c in df.columns if "T2M" in c]
    prec_cols = [c for c in df.columns if "PRECTOT" in c]

    if scenario == "baseline":
        return df

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


# ---------------- LOAD MODEL ----------------
def load_model(model_id):
    return joblib.load(f"artifacts/models/{model_id}.pkl")


# ---------------- PREDICT ----------------
def predict_risk(df, model):

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

            probs = predict_risk(df_scn, models[target])

            df_tmp = df_scn.copy()
            df_tmp["risk"] = probs

            # ✅ FIX numeric aggregation issue
            grouped = df_tmp.groupby(["Region", "Country"]).mean(numeric_only=True).reset_index()

            grouped["Scenario"] = scenario
            grouped["Target"] = target

            results.append(grouped)

    return pd.concat(results, ignore_index=True)


# ---------------- RELATIVE CHANGE ----------------
def compute_relative_change(df):

    # Extract baseline safely
    base = df[df["Scenario"] == "baseline"][["Region", "Country", "Target", "risk"]].copy()
    base = base.rename(columns={"risk": "risk_base"})

    # Merge with all scenarios
    merged = df.merge(
        base,
        on=["Region", "Country", "Target"],
        how="left"
    )

    # Prevent division errors
    merged["risk_base"] = merged["risk_base"].replace(0, np.nan)
    merged["risk_change"] = (merged["risk"] - merged["risk_base"]) / merged["risk_base"] * 100

    # Baseline = 0 change
    merged.loc[merged["Scenario"] == "baseline", "risk_change"] = 0

    return merged


# ---------------- PLOT ----------------
def plot_maps(df):

    df = compute_relative_change(df)

    scenarios = ["baseline", "ssp245", "ssp585"]
    targets = ["Aflac", "Fumc"]

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[f"{t} - {s}" for t in targets for s in scenarios],
        specs=[[{"type": "choropleth"}]*3,
               [{"type": "choropleth"}]*3],
        horizontal_spacing=0.02,
        vertical_spacing=0.05
    )

    for i, target in enumerate(targets):
        for j, scenario in enumerate(scenarios):

            data = df[(df["Target"] == target) & (df["Scenario"] == scenario)]

            if scenario == "baseline":
                z = data["risk"]
                colorscale = "Viridis"
                zmin, zmax = 0, 1
                title = "Risk"
            else:
                z = data["risk_change"]
                colorscale = "RdYlGn_r"
                zmin, zmax = -50, 100
                title = "% Change"

            fig.add_trace(
                go.Choropleth(
                    locations=data["Country"],
                    locationmode="country names",
                    z=z,
                    colorscale=colorscale,
                    zmin=zmin,
                    zmax=zmax,
                    colorbar=dict(title=title, len=0.6),
                    showscale=(j == 2)
                ),
                row=i+1,
                col=j+1
            )

    fig.update_layout(
        title="Climate Change Impact on Mycotoxin Risk (SSP Scenarios)",
        height=800,
        margin=dict(l=5, r=5, t=60, b=5)
    )

    # Cleaner maps
    for r in range(1, 3):
        for c in range(1, 4):
            fig.update_geos(
                scope="africa",
                showframe=False,
                showcoastlines=False,
                projection_type="natural earth",
                row=r,
                col=c
            )

    fig.show()


# ---------------- MAIN ----------------
if __name__ == "__main__":

    df = load_data()

    df_map = build_dataset(df)

    plot_maps(df_map)