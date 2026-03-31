import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def radar(ax, values_dict, color_map):
    metrics = ["accuracy", "precision", "recall", "f1"]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    # Axis styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)

    ax.grid(True)

    # Plot models with consistent colors
    for model, vals in values_dict.items():
        v = [vals[m] for m in metrics]
        v += v[:1]

        ax.plot(angles, v, linewidth=2, color=color_map[model])
        ax.fill(angles, v, alpha=0.1, color=color_map[model])


def create_master_figure(results_df):

    targets = ["Aflac", "Fumc"]
    feature_sets = ["weather", "weather_soil", "all"]

    models = results_df["model"].unique()

    # 🔥 Consistent color map
    cmap = plt.get_cmap("tab10")
    color_map = {model: cmap(i) for i, model in enumerate(models)}

    fig = plt.figure(figsize=(24, 18))

    # ---------------- GLOBAL TITLE ----------------
    fig.suptitle("Model Performance Summary", fontsize=18, y=0.95)

    # ---------------- TABLE ----------------
    ax_table = plt.subplot2grid((3, 3), (0, 0), colspan=3)
    ax_table.axis('off')

    table_data = results_df.round(3)

    table = ax_table.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # ---------------- RADAR GRID ----------------
    for i, target in enumerate(targets):
        for j, fs in enumerate(feature_sets):

            ax = plt.subplot2grid((3, 3), (i + 1, j), polar=True)

            subset = results_df[
                (results_df["target"] == target) &
                (results_df["feature_set"] == fs)
            ]

            values_dict = {
                row["model"]: {
                    "accuracy": row["accuracy"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"]
                }
                for _, row in subset.iterrows()
            }

            if values_dict:
                radar(ax, values_dict, color_map)

            ax.set_title(f"{target} - {fs}", fontsize=11, pad=12)

    # ---------------- GLOBAL LEGEND ----------------
    handles = [
        plt.Line2D([0], [0], color=color_map[m], lw=2)
        for m in models
    ]

    fig.legend(
        handles,
        models,
        loc="center right",
        fontsize=12,
        title="Models",
        bbox_to_anchor=(0.92, 0.5)
    )

    # ---------------- LAYOUT FIX ----------------
    plt.tight_layout(rect=[0, 0, 0.90, 0.93])  # leave space for legend

    # ---------------- SAVE ----------------
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/master_figure.png", dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()

    print("✅ Figure saved to results/plots/master_figure.png")