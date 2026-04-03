import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import joblib
import os

from sklearn.metrics import f1_score

# ---------------- STYLE ----------------
COLOR_PALETTE = sns.color_palette("tab10", 10)
sns.set_palette(COLOR_PALETTE)
sns.set_style("whitegrid")

# ---------------- LOAD DATA ----------------
def load_data():
    df = pd.read_csv("data/selected/2024pg_sfm.csv")
    df2 = pd.read_csv("data/preprocessed/2024pg.csv")

    df["Aflac"] = (df2["Afla"] > 10).astype(int)
    df["Fumc"] = (df2["Fum"] > 4000).astype(int)

    return df, df2


# ---------------- PREP ----------------
def prepare_X(df, target):
    drop_cols = ["Aflac", "Fumc", "Fum"]
    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include=np.number).fillna(0)
    return X


# ---------------- ALIGN FEATURES ----------------
def align_features(X, feature_names):
    X = X.copy()

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    return X[feature_names]


# ---------------- LOAD MODEL ----------------
def load_model(model_id):
    path = f"artifacts/models/{model_id}.pkl"

    if os.path.exists(path):
        print(f" Loaded → {model_id}")
        return joblib.load(path)

    raise FileNotFoundError(f"{model_id} not found")


# ---------------- CLIMATE SCENARIOS ----------------
def apply_climate_scenario(df, scenario):
    df = df.copy()

    climate_cols = [c for c in df.columns if "T2M" in c or "PRECTOT" in c]

    if scenario == "ssp245":
        for c in climate_cols:
            if "T2M" in c:
                df[c] += 1.5
            if "PRECTOT" in c:
                df[c] *= 1.05

    elif scenario == "ssp585":
        for c in climate_cols:
            if "T2M" in c:
                df[c] += 3.0
            if "PRECTOT" in c:
                df[c] *= 1.10

    return df


# ---------------- PREDICT ----------------
def predict_with_uncertainty(model, X, n_boot=30):
    preds = []
    probs = []

    for _ in range(n_boot):
        idx = np.random.choice(len(X), len(X), replace=True)
        X_sample = X.iloc[idx]

        p = model.predict_proba(X_sample)[:, 1]
        probs.append(p.mean())

    return np.mean(probs), np.std(probs)


# ---------------- MAP PLOTTING ----------------
def plot_map(ax, gdf, column, title, cmap="Reds"):
    gdf.plot(column=column,
             cmap=cmap,
             legend=True,
             ax=ax,
             edgecolor="black")

    ax.set_title(title)
    ax.axis("off")


# ---------------- MAIN CLIMATE FIGURE ----------------
def create_climate_figure():

    df, df2 = load_data()

    # Load shapefile (CHANGE to your path)
    gdf = gpd.read_file("data/shapefiles/nigeria_states.shp")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row, (target, model_id) in enumerate([
        ("Aflac", "allafrf"),
        ("Fumc", "allfurf")
    ]):

        model = load_model(model_id)
        feature_names = model.named_steps["scaler"].feature_names_in_

        # ---------------- CURRENT ----------------
        X_current = prepare_X(df, target)
        X_current = align_features(X_current, feature_names)

        current_mean, current_std = predict_with_uncertainty(model, X_current)

        # ---------------- SCENARIOS ----------------
        df_245 = apply_climate_scenario(df, "ssp245")
        df_585 = apply_climate_scenario(df, "ssp585")

        X_245 = align_features(prepare_X(df_245, target), feature_names)
        X_585 = align_features(prepare_X(df_585, target), feature_names)

        mean_245, std_245 = predict_with_uncertainty(model, X_245)
        mean_585, std_585 = predict_with_uncertainty(model, X_585)

        # ---------------- RELATIVE CHANGE ----------------
        change_245 = (mean_245 - current_mean) / current_mean * 100
        change_585 = (mean_585 - current_mean) / current_mean * 100

        # Attach to map (uniform for demo — replace with spatial aggregation later)
        gdf[f"{target}_current"] = current_mean
        gdf[f"{target}_245"] = change_245
        gdf[f"{target}_585"] = change_585

        # ---------------- PLOTS ----------------
        plot_map(
            axes[row, 0],
            gdf,
            f"{target}_current",
            f"{target} Current Risk\n(mean={current_mean:.2f})"
        )

        plot_map(
            axes[row, 1],
            gdf,
            f"{target}_245",
            f"{target} Δ Risk SSP2-4.5\n({change_245:.1f}%)",
            cmap="coolwarm"
        )

        plot_map(
            axes[row, 2],
            gdf,
            f"{target}_585",
            f"{target} Δ Risk SSP5-8.5\n({change_585:.1f}%)",
            cmap="coolwarm"
        )

    fig.suptitle("Climate Change Impact on Mycotoxin Risk", fontsize=18)

    os.makedirs("results/plots", exist_ok=True)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig("results/plots/figure5_climatechange.png", dpi=300)

    plt.show()


# ---------------- RUN ----------------
if __name__ == "__main__":
    create_climate_figure()