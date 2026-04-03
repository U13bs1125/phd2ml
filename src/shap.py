import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import os
import glob

# ---------------- LOAD TOP FEATURE FILES ----------------
def load_shap_files(shap_dir="artifacts/shap"):
    """
    Returns dict: {(target, feature_set): {model_name: shap_df}}
    Expects CSV naming: {feature_set}_{target}_{model}_top10.csv
    """
    shap_files = glob.glob(os.path.join(shap_dir, "*.csv"))
    shap_data = {}

    for f in shap_files:
        name = os.path.basename(f).replace(".csv", "")
        parts = name.split("_")
        if len(parts) < 3:
            continue

        # Parse from the end
        if parts[-1] == "top10":
            parts = parts[:-1]  # remove 'top10'
        model_name = parts[-1]       # last part is model
        target = parts[-2]           # second last is target
        feature_set = "_".join(parts[:-2])  # the rest is feature set

        key = (target, feature_set)
        if key not in shap_data:
            shap_data[key] = {}
        shap_data[key][model_name] = pd.read_csv(f)

    return shap_data

# ---------------- PLOT ----------------
def plot_shap_grid():
    shap_data = load_shap_files()

    # ---------------- DEFINE ORDER ----------------
    targets = ["Aflac", "Fumc"]
    feature_sets = ["weather", "weather_soil", "weather_soil_agro"]
    model_names = ["nn", "rf", "xgb", "ls"]  # 4 columns

    n_rows = len(targets) * len(feature_sets)  # 6 rows
    n_cols = len(model_names)  # 4 columns

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    # Fix axes shape
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    # ---------------- LOOP ----------------
    for i, target in enumerate(targets):
        for j, fs_name in enumerate(feature_sets):
            row_idx = i * len(feature_sets) + j
            key = (target, fs_name)
            for col_idx, model_name in enumerate(model_names):
                ax = axes[row_idx][col_idx]
                shap_df = shap_data.get(key, {}).get(model_name)

                if shap_df is None or shap_df.empty:
                    # show placeholder subplot
                    ax.text(0.5, 0.5, "No data", ha='center', va='center')
                    ax.axis("off")
                    continue

                # ---------------- HANDLE ALL-ZERO SHAP ----------------
                if (shap_df['shap_value'].abs().sum() == 0):
                    shap_df_plot = shap_df.copy()
                    # Add tiny dummy value to force bar plot
                    shap_df_plot.loc[0, 'shap_value'] = 1e-6
                else:
                    shap_df_plot = shap_df

                # ---------------- BAR PLOT ----------------
                ax.barh(shap_df_plot['feature'][::-1], shap_df_plot['shap_value'][::-1], color='skyblue')
                ax.set_xlabel("Mean |SHAP| / Importance")
                ax.set_title(f"{target} - {fs_name} - {model_name}", fontsize=9)

    fig.suptitle("Top 10 Feature Importance per Model / Feature Set", fontsize=16, y=0.95)

    os.makedirs("results/plots/shap", exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("results/plots/shap/shap_top10_grid_4x6.png", dpi=300)

    plt.close()

    print("✅ SHAP/Feature Importance top-10 grid saved → results/plots/shap/shap_top10_grid_4x6.png")


# ---------------- RUN ----------------
if __name__ == "__main__":
    plot_shap_grid()