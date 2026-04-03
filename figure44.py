import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler

# ---------------- STYLE ----------------
COLOR_PALETTE = sns.color_palette("tab10", 10)
sns.set_palette(COLOR_PALETTE)
sns.set_style("whitegrid")

scaler = StandardScaler()

# ---------------- LOAD DATA ----------------
def load_data(file="data/processed/2024pg.csv"):
    df = pd.read_csv(file)
    df["Aflac"] = np.where(df["Afla"] > 10, 1, 0)
    return df

def load_data2(file="data/preprocessed/2024pg.csv"):
    df2 = pd.read_csv(file)

    for col in ["Harvestdate", "Sowdate"]:
        if col in df2.columns:
            df2[col] = pd.to_datetime(df2[col], errors="coerce")

    df2["Harvest_year"] = df2["Harvestdate"].dt.year
    df2["Aflac"] = np.where(df2["Afla"] > 10, 1, 0)

    categorical_cols = [
        "Color","Tillage","Biocide","Fertilizer",
        "Seedprep","Awareness","Sowmethod","Prevtime","Prevcrop"
    ]
    categorical_cols = [c for c in categorical_cols if c in df2.columns]

    df2 = pd.get_dummies(df2, columns=categorical_cols, drop_first=True)

    bool_cols = df2.select_dtypes(include=["bool"]).columns
    df2[bool_cols] = df2[bool_cols].astype(int)

    drop_cols = ["Id","Harvestdate","Sowdate","Longitude","Latitude","Region"]
    df2.drop(columns=[c for c in drop_cols if c in df2.columns], inplace=True)

    return df2

# ---------------- PREP ----------------
def prepare_X_y(df):
    drop_cols = ["Afla", "Aflac", "Fumc"]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["Aflac"]
    X = X.select_dtypes(include=np.number).fillna(0)
    return X, y

# ---------------- TRAIN ----------------
def train_model(X, y, model_type="rf"):
    X_scaled = scaler.fit_transform(X)

    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    elif model_type == "rf_small":
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    elif model_type == "logreg":
        model = LogisticRegression(max_iter=2000)
    else:
        raise ValueError("Unknown model type")

    model.fit(X_scaled, y)
    return model

# ---------------- BOOTSTRAP ----------------
def bootstrap_metric(y_true, y_pred, n_boot=100):
    scores = []
    n = len(y_true)

    for _ in range(n_boot):
        idx = np.random.choice(range(n), n, replace=True)
        scores.append(f1_score(y_true.iloc[idx], y_pred[idx]))

    return np.mean(scores), np.std(scores)

# ---------------- CLIMATE SIMILARITY ----------------
def climate_similarity(train_df, test_df):
    weather_cols = [c for c in train_df.columns if "T2M" in c or "PRECTOT" in c or "RH2M" in c]

    if len(weather_cols) == 0:
        return 1

    dist = np.linalg.norm(train_df[weather_cols].mean() - test_df[weather_cols].mean())
    return np.exp(-dist)

# ---------------- PANEL A ----------------
def plot_temporal(ax, df):
    df2 = load_data2()
    years = sorted(df2["Harvest_year"].unique())
    means, stds = [], []

    for i in range(len(years)-1):
        train = df[df2["Harvest_year"] <= years[i]]
        test = df[df2["Harvest_year"] == years[i+1]]

        X_train, y_train = prepare_X_y(train)
        X_test, y_test = prepare_X_y(test)

        model = train_model(X_train, y_train)
        preds = model.predict(X_test)

        mean, std = bootstrap_metric(y_test, preds)
        means.append(mean)
        stds.append(std)

    x = years[1:]
    ax.plot(x, means, marker="o")
    ax.fill_between(x, np.array(means)-np.array(stds),
                    np.array(means)+np.array(stds), alpha=0.2)

    ax.set_title("A. Temporal Validation")
    ax.set_xlabel("Test Year")
    ax.set_ylabel("F1-score")
    ax.set_xlim(2023, 2027)

# ---------------- PANEL B ----------------
def plot_spatial_heatmap(ax, df):
    df2 = load_data2()
    countries = df2["Country"].unique()
    matrix = np.zeros((len(countries), len(countries)))

    for i, train_c in enumerate(countries):
        for j, test_c in enumerate(countries):

            train = df[df2["Country"] == train_c]
            test = df[df2["Country"] == test_c]

            if len(train) < 20 or len(test) < 20:
                matrix[i, j] = np.nan
                continue

            X_train, y_train = prepare_X_y(train)
            X_test, y_test = prepare_X_y(test)

            model = train_model(X_train, y_train)
            preds = model.predict(scaler.transform(X_test))

            matrix[i, j] = f1_score(y_test, preds)

    sns.heatmap(matrix, xticklabels=countries,
                yticklabels=countries, cmap="viridis", ax=ax)

    ax.set_title("B. Spatial Transfer")

# ---------------- PANEL C ----------------
def plot_generalization_robustness(ax, df):
    df2 = load_data2()

    train = df[df2["Harvest_year"] < 2025]
    test = df[df2["Harvest_year"] == 2025]
    nigeria = df[df2["Country"].str.lower().str.strip() == "nigeria"]

    X_train, y_train = prepare_X_y(train)
    X_test, y_test = prepare_X_y(test)
    X_ext, y_ext = prepare_X_y(nigeria)

    model = train_model(X_train, y_train)

    scores, stds = [], []

    for X_, y_ in [(X_train,y_train),(X_test,y_test),(X_ext,y_ext)]:
        preds = model.predict(scaler.transform(X_))
        m, s = bootstrap_metric(y_, preds)
        scores.append(m)
        stds.append(s)

    base = scores[0]
    drop_test = (base - scores[1]) / base * 100
    drop_ext = (base - scores[2]) / base * 100

    similarity = climate_similarity(train, nigeria)
    norm_drop = drop_ext * (1 - similarity)

    labels = ["Train", "Test", "Nigeria"]

    ax.bar(labels, scores)
    ax.errorbar(labels, scores, yerr=stds, fmt="none", capsize=5)

    ax.set_title(
        f"C. Generalization & Robustness\n"
        f"Test drop={drop_test:.1f}% | Nigeria drop={drop_ext:.1f}% | Norm={norm_drop:.1f}%"
    )

# ---------------- PANEL D ----------------
def plot_calibration(ax, df):
    df2 = load_data2()

    train = df[df2["Harvest_year"] < 2025]
    test = df[df2["Harvest_year"] == 2025]

    X_train, y_train = prepare_X_y(train)
    X_test, y_test = prepare_X_y(test)

    model = train_model(X_train, y_train)
    probs = model.predict_proba(scaler.transform(X_test))[:, 1]

    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)

    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.plot([0,1],[0,1], linestyle="--", label="Perfect")

    ax.set_title("D. Calibration Curve")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend()

# ---------------- PANEL E ----------------
def plot_model_perf(ax, df):
    X, y = prepare_X_y(df)

    models = ["rf", "rf_small", "logreg"]
    labels = ["RF (200)", "RF (50)", "LogReg"]

    scores, stds = [], []

    for m in models:
        model = train_model(X, y, model_type=m)
        preds = model.predict(scaler.transform(X))
        mean, std = bootstrap_metric(y, preds)

        scores.append(mean)
        stds.append(std)

    ax.bar(labels, scores)
    ax.errorbar(labels, scores, yerr=stds, fmt="none", capsize=5)

    ax.set_title("E. Model Comparison (CI)")
    ax.set_ylabel("F1-score")

# ---------------- PANEL F ----------------
def plot_crop_vs_generic(ax, df):
    df2 = load_data2()

    X, y = prepare_X_y(df)
    generic = train_model(X, y)

    maize_df = df[df2["Crop"].str.lower().str.strip() == "maize"]
    Xc, yc = prepare_X_y(maize_df)

    crop_model = train_model(Xc, yc)

    g_mean, g_std = bootstrap_metric(yc, generic.predict(scaler.transform(Xc)))
    c_mean, c_std = bootstrap_metric(yc, crop_model.predict(scaler.transform(Xc)))

    ax.bar(["Generic", "Maize"], [g_mean, c_mean])
    ax.errorbar(["Generic","Maize"],
                [g_mean,c_mean],
                yerr=[g_std,c_std],
                fmt="none", capsize=5)

    ax.set_title("F. Crop vs Generic")

# ---------------- MAIN ----------------
def create_figure(df):

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    plot_temporal(axes[0], df)
    plot_spatial_heatmap(axes[1], df)
    plot_generalization_robustness(axes[2], df)

    plot_calibration(axes[3], df)
    plot_model_perf(axes[4], df)
    plot_crop_vs_generic(axes[5], df)

    fig.suptitle("Figure: Spatial & Temporal Generalization", fontsize=18)

    os.makedirs("results/plots", exist_ok=True)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig("results/plots/figure4_generalization.png", dpi=300)

    plt.show()

# ---------------- RUN ----------------
if __name__ == "__main__":
    df = load_data("data/processed/2024pg.csv")
    create_figure(df)