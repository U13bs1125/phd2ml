import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point

# Color palette for consistent styling
COLOR_PALETTE = sns.color_palette("tab10", 10)
sns.set_palette(COLOR_PALETTE)

# ---------------- WORKFLOW ----------------
def plot_workflow(ax):
    # Flowchart-style workflow
    workflow_data = {
        "Data": ["Survey data", "Rapid test", "NASA", "SoilGrid"],
        "Features": ["Weather", "Soil", "Phenology", "Agro"],
        "Models": ["Neural Net", "RF", "GB", "Lasso"],
        "Prediction": ["Afla", "Fum"],
        "Climate": ["2030 RCP3.5", "2030 RCP8.5", "2050 RCP3.5", "2050 RCP8.5"]
    }
    
    steps = list(workflow_data.keys())
    
    # Colors for flowchart
    colors = ["#E8F4F8", "#B3E5FC", "#81D4FA", "#4FC3F7", "#29B6F6"]
    
    # Draw flowchart boxes
    box_width = 0.75
    box_height = 0.25
    
    for i, (step, color) in enumerate(zip(steps, colors)):
        x = i * 1.55
        y = 0.7
        
        # Main step box with border
        rect = plt.Rectangle((x - box_width/2, y - box_height/2), box_width, box_height,
                             facecolor=color, edgecolor='#01579B', linewidth=2)
        ax.add_patch(rect)
        
        # Step title
        ax.text(x, y, step, ha='center', va='center', fontsize=9, fontweight='bold', color='#01579B')
        
        # Subsections below
        subsections = workflow_data[step]
        n_sub = len(subsections)
        y_start = 0.38
        for j, sub in enumerate(subsections):
            y_offset = y_start - (j * 0.12)
            ax.text(x, y_offset, f"• {sub}", ha='center', va='center', fontsize=6.5, color='#01579B')
        
        # Connecting arrows
        if i < len(steps) - 1:
            arrow = plt.Arrow(x + box_width/2 + 0.05, y, 0.35, 0, 
                            width=0.06, color='#01579B')
            ax.add_patch(arrow)
    
    ax.set_xlim(-0.6, 8.5)
    ax.set_ylim(-0.1, 1.0)
    ax.axis("off")
    ax.set_title("A. Workflow", fontsize=10, fontweight='bold')

# ---------------- GEO MAP ----------------
def plot_geo_map(ax, df, group, title):
    if {"Latitude", "Longitude", group}.issubset(df.columns):
        world = gpd.read_file('https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip')
        world.plot(ax=ax, color="lightgrey", edgecolor="white")

        geometry = [Point(xy) for xy in zip(df["Longitude"], df["Latitude"])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry)

        groups = df[group].dropna().unique()
        palette = sns.color_palette("tab10", len(groups))
        color_dict = {g: palette[i] for i, g in enumerate(groups)}

        for g in groups:
            subset = gdf[gdf[group] == g]
            subset.plot(ax=ax, color=color_dict[g], markersize=15, alpha=0.7, label=str(g))
        ax.legend(fontsize=6)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    else:
        ax.axis("off")
        ax.text(0.5,0.5,"No Map Data", ha='center')

# ---------------- DISTRIBUTION ----------------
def plot_distribution(ax, df, target):
    if {"Country","Crop",target}.issubset(df.columns):
        sns.boxplot(x='Country', y=target, hue='Crop', data=df, ax=ax, palette=COLOR_PALETTE[:3])
        ax.set_yscale('symlog', linthresh=0.1)
        ax.set_title(f"{target} Distribution", fontsize=10)
        ax.legend(fontsize=6)
    else:
        ax.axis("off")
        ax.text(0.5,0.5,"No Distribution Data", ha='center')

# ---------------- CLASS IMBALANCE ----------------
def plot_imbalance(ax, df):
    if {"Afla","Fum"}.issubset(df.columns):
        targets = {"Aflac": (df["Afla"] > 10).astype(int),
                   "Fumc": (df["Fum"] > 4000).astype(int)}
        width = 0.35
        labels = ["0","1"]
        x = np.arange(len(labels))
        for i, (name, arr) in enumerate(targets.items()):
            ax.bar(x + i*width, np.bincount(arr, minlength=2), width, label=name, color=COLOR_PALETTE[i])
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(labels)
        ax.set_title("Class Imbalance", fontsize=10)
        ax.legend(fontsize=6)
    else:
        ax.axis("off")
        ax.text(0.5,0.5,"No Class Imbalance Data", ha='center')

# ---------------- FEATURE GROUPS ----------------
def plot_feature_groups(ax, feature_dict):
    groups = ["weather", "soil", "agro"]
    counts = [len(feature_dict.get(g, [])) for g in groups]
    colors = [COLOR_PALETTE[:3][i] for i in range(len(groups))]
    ax.bar(groups, counts, color=colors)
    ax.set_title("Feature Groups", fontsize=10)

# ---------------- SUMMARY TABLE ----------------
def plot_summary_table(ax, df):
    if {"Country","Region","Afla","Fum"}.issubset(df.columns):
        df = df.copy()
        df["afla_pos"] = (df["Afla"] > 0).astype(int)
        df["fum_pos"] = (df["Fum"] > 0).astype(int)
        df["afla>thr(10)"] = (df["Afla"] > 10).astype(int)
        df["fum_>thr(4000)"] = (df["Fum"] > 4000).astype(int)

        summary = df.groupby(["Country","Region"]).agg(
            afla_pos=("afla_pos","sum"),
            fum_pos=("fum_pos","sum"),
            afla_thr=("afla>thr(10)","sum"),
            fum_thr=("fum_>thr(4000)","sum"),
            total=("Afla","count")
        ).reset_index()

        for col in ["afla_pos","fum_pos","afla_thr","fum_thr"]:
            summary[f"{col}_%"] = summary[col] / summary["total"]

        ax.axis("off")
        
        table = ax.table(
            cellText=summary.round(2).values,
            colLabels=summary.columns,
            loc="center",
            cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(5)  # smaller font
        table.scale(1, 1.2)
    else:
        ax.axis("off")
        ax.text(0.5,0.5,"No Summary Data", ha='center')

# ---------------- EXPLANATION BOX ----------------
def plot_explanation(fig):
    """Add figure explanation text at the bottom"""
    explanation_text = (
        "Panel Descriptions:\n"
        "A. Workflow: Five-step data analysis pipeline from data collection through climate change predictions.\n"
        "B. Country Map: Geographic distribution of samples across countries with color-coded markers.\n"
        "C. Crop Map: Geographic distribution of samples by crop type across the study region.\n"
        "D. Summary Table: Aggregated statistics by country and region showing contamination frequencies and prevalence.\n"
        "E. Feature Groups: Count of features across three categories (weather, soil, agronomic).\n"
        "F. Afla Distribution: Distribution of aflatoxin contamination levels across countries and crops (log scale).\n"
        "G. Fum Distribution: Distribution of fumonisin contamination levels across countries and crops (log scale).\n"
        "H. Class Imbalance: Binary classification balance for aflatoxin and fumonisin contamination thresholds."
    )
    
    fig.text(
        0.85, 0.01, explanation_text,
        ha='center', va='bottom',
        fontsize=7, wrap=True,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, pad=1)
    )

# ---------------- MAIN FIGURE ----------------
def create_figure(df, feature_dict):
    fig, axes = plt.subplots(2,4, figsize=(22,15))
    axes = axes.flatten()

    # Row 1
    plot_workflow(axes[0])
    plot_geo_map(axes[1], df, "Country", "Country Map")
    #plot_geo_map(axes[2], df, "Region", "Region Map")
    plot_geo_map(axes[2], df, "Crop", "Crop Map")
    plot_summary_table(axes[3], df)

    # Row 2
    plot_feature_groups(axes[4], feature_dict)
    plot_distribution(axes[5], df, "Afla")
    plot_distribution(axes[6], df, "Fum")
    plot_imbalance(axes[7], df)

    
    

    fig.suptitle("Figure 1: Data, Environment & Risk Structure", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    
    # Add explanation box
    plot_explanation(fig)

    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/figure1.png", dpi=300)
    plt.close()

# ---------------- RUN ----------------
if __name__ == "__main__":
    from src.preprocessing import preprocess_data
    dff, feature_dict = preprocess_data("data/preprocessed/2024pg.csv")
    df = pd.read_csv("data/preprocessed/2024pg.csv")
    create_figure(df, feature_dict)