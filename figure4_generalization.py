import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.calibration import calibration_curve

from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# ---------------- STYLE ----------------
COLOR_PALETTE = sns.color_palette("tab10", 10)
sns.set_palette(COLOR_PALETTE)
sns.set_style("whitegrid")

# ---------------- LOAD DATA ----------------
def load_data():
    df = pd.read_csv("data/selected/2024pg_sfm.csv")
    df2 = pd.read_csv("data/preprocessed/2024pg.csv")

    df["Aflac"] = np.where(df2["Afla"] > 10, 1, 0)
    df["Fumc"] = np.where(df2["Fum"] > 4000, 1, 0)

    return df

def load_data2():
    df2 = pd.read_csv("data/preprocessed/2024pg.csv")

    df2["Harvestdate"] = pd.to_datetime(df2["Harvestdate"], errors="coerce")
    df2["Harvest_year"] = df2["Harvestdate"].dt.year

    df2["Aflac"] = np.where(df2["Afla"] > 10, 1, 0)
    df2["Fumc"] = np.where(df2["Fum"] > 4000, 1, 0)

    return df2.reset_index(drop=True)

# ---------------- PREP ----------------
def prepare_X_y(df, target):
    drop_cols = ["Aflac", "Fumc", "Fum"]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target]

    X = X.select_dtypes(include=np.number).fillna(0)
    return X, y

# ---------------- FEATURE ALIGN ----------------
def align_features(X, reference_features):
    X = X.copy()

    for col in reference_features:
        if col not in X.columns:
            X[col] = 0

    X = X[reference_features]
    return X

# ---------------- LOAD MODEL ----------------
def load_model_pipeline(model_id):
    path = f"artifacts/models/{model_id}.pkl"

    if os.path.exists(path):
        print(f" Loaded → {model_id}")
        return joblib.load(path)
    return None

# ---------------- TRAIN PIPELINE ----------------
def train_pipeline(X, y):
    minority_count = min(y.value_counts())
    k_neighbors = min(3, minority_count - 1) if minority_count > 1 else 0

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(k_neighbors=k_neighbors, random_state=42) if k_neighbors > 0 else None),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    pipe.steps = [(n, s) for n, s in pipe.steps if s is not None]

    pipe.fit(X, y)
    return pipe

# ---------------- BOOTSTRAP ----------------
def bootstrap_metric(y_true, y_pred, n_boot=100):
    scores = []
    n = len(y_true)

    for _ in range(n_boot):
        idx = np.random.choice(range(n), n, replace=True)
        scores.append(f1_score(y_true.iloc[idx], y_pred[idx], zero_division=0))

    return np.mean(scores), np.std(scores)

# ---------------- CLIMATE SIMILARITY ----------------
def climate_similarity(train_df, test_df):
    weather_cols = [c for c in train_df.columns if "T2M" in c or "PRECTOT" in c]

    if len(weather_cols) == 0:
        return 1

    dist = np.linalg.norm(train_df[weather_cols].mean() - test_df[weather_cols].mean())
    return np.exp(-dist)

# ---------------- PANEL A ----------------
def plot_temporal(ax, df):
    df2 = load_data2()
    years = sorted(df2["Harvest_year"].dropna().unique())

    for target, model_id, color in zip(
        ["Aflac", "Fumc"],
        ["allafrf", "allfurf"],
        COLOR_PALETTE[:2]
    ):
        model = load_model_pipeline(model_id)
        if model is None:
            continue

        features = model.named_steps["scaler"].feature_names_in_

        means, stds = [], []

        for i in range(len(years)-1):
            test = df[df2["Harvest_year"] == years[i+1]]

            X_test, y_test = prepare_X_y(test, target)
            X_test = align_features(X_test, features)

            preds = model.predict(X_test)

            m, s = bootstrap_metric(y_test, preds)
            means.append(m)
            stds.append(s)

        x = years[1:len(means)+1]

        ax.plot(x, means, marker="o", label=target, color=color)
        ax.fill_between(x,
                        np.array(means)-np.array(stds),
                        np.array(means)+np.array(stds),
                        alpha=0.2)

    ax.set_xlim(2023, 2027)
    ax.set_title("A. Temporal Validation")
    ax.legend()

# ---------------- PANEL B ----------------
def plot_spatial_heatmap(ax, df, target):

    df2 = load_data2()
    countries = df2["Country"].dropna().unique()

    matrix = np.zeros((len(countries), len(countries)))

    for i, train_c in enumerate(countries):
        for j, test_c in enumerate(countries):

            train = df[df2["Country"] == train_c]
            test = df[df2["Country"] == test_c]

            if len(train) < 20 or len(test) < 20:
                matrix[i, j] = np.nan
                continue

            X_train, y_train = prepare_X_y(train, target)
            X_test, y_test = prepare_X_y(test, target)

            model = train_pipeline(X_train, y_train)
            X_test = align_features(X_test, X_train.columns)

            preds = model.predict(X_test)
            matrix[i, j] = f1_score(y_test, preds, zero_division=0)

    sns.heatmap(matrix,
                xticklabels=countries,
                yticklabels=countries,
                cmap="viridis",
                ax=ax)

    ax.set_title(f"B. Spatial Transfer ({target})")

# ---------------- PANEL C (FIXED WITH CI) ----------------
def plot_generalization(ax, df):

    df2 = load_data2()

    train = df[df2["Harvest_year"] < 2025]
    test = df[df2["Harvest_year"] == 2025]
    nigeria = df[df2["Country"].str.lower() == "nigeria"]

    width = 0.35

    for i, (target, model_id) in enumerate([
        ("Aflac", "allafrf"),
        ("Fumc", "allfurf")
    ]):

        model = load_model_pipeline(model_id)
        features = model.named_steps["scaler"].feature_names_in_

        scores, stds = [], []

        for subset in [train, test, nigeria]:
            X_, y_ = prepare_X_y(subset, target)
            X_ = align_features(X_, features)

            preds = model.predict(X_)
            m, s = bootstrap_metric(y_, preds)

            scores.append(m)
            stds.append(s)

        base = scores[0]

        drop_ext = (base - scores[2]) / base * 100 if base > 0 else 0
        sim = climate_similarity(train, nigeria)
        norm_drop = drop_ext * (1 - sim)

        pos = np.arange(3) + i*width

        ax.bar(pos, scores, width=width, label=target)
        ax.errorbar(pos, scores, yerr=stds, fmt="none", capsize=5)

        ax.text(pos[2], scores[2]+0.02,
                f"drop={drop_ext:.1f}%\nnorm={norm_drop:.1f}%",
                fontsize=8)

    ax.set_xticks(np.arange(3) + width/2)
    ax.set_xticklabels(["Train", "Test", "Nigeria"])
    ax.set_title("C. Generalization + Robustness (CI)")
    ax.set_ylabel("F1-score")
    ax.legend()

# ---------------- PANEL D ----------------
def plot_calibration(ax, df):
    df2 = load_data2()
    test = df[df2["Harvest_year"] == 2025]

    for target, model_id, color in zip(
        ["Aflac", "Fumc"],
        ["allafrf", "allfurf"],
        COLOR_PALETTE[:2]
    ):
        model = load_model_pipeline(model_id)
        features = model.named_steps["scaler"].feature_names_in_

        X_test, y_test = prepare_X_y(test, target)
        X_test = align_features(X_test, features)

        probs = model.predict_proba(X_test)[:, 1]

        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)

        ax.plot(mean_pred, frac_pos, marker="o", label=target)

    ax.plot([0,1],[0,1],"--", label="Perfect")
    ax.set_title("D. Calibration")
    ax.legend()

# ---------------- PANEL E ----------------
def plot_crop_vs_generic(ax, df):
    df2 = load_data2()

    maize = df[df2["Crop"].str.lower() == "maize"]
    width = 0.35

    for i, (target, model_id) in enumerate([
        ("Aflac", "allafrf"),
        ("Fumc", "allfurf")
    ]):

        generic = load_model_pipeline(model_id)

        Xc, yc = prepare_X_y(maize, target)

        crop_model = train_pipeline(Xc, yc) if len(Xc) > 50 else RandomForestClassifier().fit(Xc, yc)

        features = generic.named_steps["scaler"].feature_names_in_
        Xc_aligned = align_features(Xc, features)

        g_mean, g_std = bootstrap_metric(yc, generic.predict(Xc_aligned))
        c_mean, c_std = bootstrap_metric(yc, crop_model.predict(Xc))

        pos = np.arange(2) + i*width

        ax.bar(pos, [g_mean, c_mean], width=width, label=target)
        ax.errorbar(pos, [g_mean, c_mean],
                    yerr=[g_std, c_std], fmt="none", capsize=5)

    ax.set_xticks(np.arange(2) + width/2)
    ax.set_xticklabels(["Generic", "Maize"])
    ax.set_title("E. Crop vs Generic (CI)")
    ax.legend()

# ---------------- MAIN ----------------
def create_generalization_figure(df):

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    plot_temporal(axes[0,0], df)
    plot_spatial_heatmap(axes[0,1], df, "Aflac")
    plot_spatial_heatmap(axes[0,2], df, "Fumc")

    plot_generalization(axes[1,0], df)
    plot_calibration(axes[1,1], df)
    plot_crop_vs_generic(axes[1,2], df)

    plt.tight_layout()
    plt.savefig("results/plots/figure4_generalization.png", dpi=300)


# ---------------- RUN ----------------
if __name__ == "__main__":
    df = load_data()
    create_generalization_figure(df)