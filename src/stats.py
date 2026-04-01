import pandas as pd
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import os
import numpy as np

# ---------------- LOAD RESULTS ----------------
def load_metrics(file_path="results/metrics.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metrics file not found: {file_path}")
    return pd.read_csv(file_path)

# ---------------- SUMMARY STATS ----------------
def summarize_models(metrics_df):
    """
    Compute mean and std per model/target/feature_set
    """
    summary = metrics_df.groupby(["target", "feature_set", "model_id", "model"]).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
    ).reset_index()
    
    # Save summary
    os.makedirs("results/stats", exist_ok=True)
    summary.to_csv("results/stats/summary_metrics.csv", index=False)
    
    return summary

# ---------------- BEST MODEL ----------------
def get_best_model(metrics_df, metric="f1"):
    """
    Return the model_id with the highest mean metric per target/feature_set
    """
    summary = summarize_models(metrics_df)
    best_models = summary.loc[summary.groupby(["target", "feature_set"])[f"{metric}_mean"].idxmax()]
    return best_models.reset_index(drop=True)

# ---------------- MODEL COMPARISON ----------------
def compare_models(metrics_df, model1_id, model2_id, metric="f1"):
    """
    Paired t-test between two models' metric values
    """
    m1 = metrics_df[metrics_df["model_id"] == model1_id][metric]
    m2 = metrics_df[metrics_df["model_id"] == model2_id][metric]

    min_len = min(len(m1), len(m2))
    if min_len == 0:
        return {"t_stat": np.nan, "p_value": np.nan}
    
    m1 = m1.iloc[:min_len]
    m2 = m2.iloc[:min_len]

    if m1.std() == 0 or m2.std() == 0:
        return {"t_stat": np.nan, "p_value": np.nan}

    stat, p = ttest_rel(m1, m2)
    return {"t_stat": stat, "p_value": p}

# ---------------- PLOT BEST MODELS (dynamic vertical) ----------------
def plot_best_models(best_df, metric="f1"):
    """
    Plot best models per target/feature_set as vertical bars,
    grounded at zero, dynamic number of rows.
    """
    if best_df.empty:
        print("❌ No best models to plot")
        return

    n_rows = len(best_df)
    fig, axes = plt.subplots(n_rows, 1, figsize=(6, 2.5 * n_rows), sharex=True)

    if n_rows == 1:
        axes = [axes]

    for i, (_, row) in enumerate(best_df.iterrows()):
        ax = axes[i]
        mean_val = row[f"{metric}_mean"]
        std_val = row[f"{metric}_std"] if not np.isnan(row[f"{metric}_std"]) else 0

        ax.bar([0], [mean_val], yerr=[std_val], color='skyblue', capsize=5)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_ylabel(f"{row['target']} - {row['feature_set']}", rotation=0, labelpad=60, va="center", fontsize=10)
        ax.set_title(f"{row['model']} ({row['model_id']})", fontsize=11)

    fig.suptitle(f"Best Models per Target & Feature Set ({metric.upper()} ± STD)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs("results/plots/stats", exist_ok=True)
    save_path = f"results/plots/best_models_{metric}_dynamic.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Best model plot saved → {save_path}")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    metrics_df = load_metrics()
    best_models_df = get_best_model(metrics_df)
    plot_best_models(best_models_df, metric="f1")

    