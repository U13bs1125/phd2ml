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
    steps = ["Data", "Features", "Models", "Prediction", "Climate change"]
    for i, s in enumerate(steps):
        ax.text(i, 0.5, s, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        if i < len(steps) - 1:
            ax.arrow(i + 0.35, 0.5, 0.3, 0, head_width=0.05, head_length=0.05, fc='k', ec='k')
    ax.axis("off")
    ax.set_title("A. Workflow", fontsize=10)

# ---------------- GEO MAP ----------------
def plot_geo_map(ax, df, group, title, hatch_group=None, use_country=False):

    required_cols = {"Latitude", "Longitude", group}
    if not required_cols.issubset(df.columns):
        ax.axis("off")
        ax.text(0.5, 0.5, "No Map Data", ha='center')
        return

    # Load map
    world = gpd.read_file("data/Africa_shapefiles/Africa_Countries.shp")

    # -------------------------------
    # COUNTRY-LEVEL MODE (COLOR + HATCH)
    # -------------------------------
    if use_country and "Country" in df.columns:

        # Clean names
        df["Country"] = df["Country"].str.strip().str.upper()
        world["Country"] = world["Country"].str.strip().str.upper()
        print(world[[group, hatch_group]].isna().sum())

        # Aggregate dominant values
        agg_dict = {
            group: lambda x: x.mode()[0] if not x.mode().empty else None
        }

        if hatch_group and hatch_group in df.columns:
            agg_dict[hatch_group] = lambda x: x.mode()[0] if not x.mode().empty else None

        agg = df.groupby("Country").agg(agg_dict).reset_index()
        world = world.merge(agg, on="Country", how="left")

        # Color palette
        groups = world[group].dropna().unique()
        palette = sns.color_palette("tab10", len(groups))
        color_dict = {g: palette[i] for i, g in enumerate(groups)}

        # Hatch patterns
        hatch_patterns = ["///", "...", "xxx", "\\\\", "++"]
        hatch_dict = {}
        if hatch_group:
            hatch_vals = world[hatch_group].dropna().unique()
            hatch_dict = {h: hatch_patterns[i % len(hatch_patterns)] for i, h in enumerate(hatch_vals)}

        # Draw countries
        for _, row in world.iterrows():
            color = color_dict.get(row[group], "lightgrey")
            hatch = hatch_dict.get(row[hatch_group], "") if hatch_group else ""

            gpd.GeoSeries([row.geometry]).plot(
                ax=ax,
                facecolor=color,
                edgecolor="black",
                hatch=hatch,
                linewidth=0.5
            )

    else:
        # fallback: plain map
        world.plot(ax=ax, color="lightgrey", edgecolor="white")

    # -------------------------------
    # POINT PLOTTING (your original logic)
    # -------------------------------
    geometry = [Point(xy) for xy in zip(df["Longitude"], df["Latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    gdf = gdf.to_crs(world.crs)

    groups = df[group].dropna().unique()
    palette = sns.color_palette("tab10", len(groups))
    color_dict = {g: palette[i] for i, g in enumerate(groups)}

    for g in groups:
        subset = gdf[gdf[group] == g]
        subset.plot(
            ax=ax,
            color=color_dict[g],
            markersize=2,
            alpha=0.6,
            label=str(g)
        )

    ax.legend(fontsize=6)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()

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

# PLOT correlations
def plot_correlations(ax, df, cols, title):
    cols = [c for c in cols if c in df.columns]

    if len(cols) > 1:
        corr = df[cols].corr()
        sns.heatmap(corr, ax=ax, cmap="coolwarm", annot=False)
        ax.set_title(title, fontsize=10)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Not enough data", ha="center")
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

df3 = pd.read_csv("data/ari_values.csv")
import json    
soil_cols = [c for c in json.load(open("artifacts/features/soil.json"))] + ["Afla"]

temp_cols = [c for c in json.load(open("artifacts/features/weather.json"))] + ["Afla"]
rh_cols = [c for c in json.load(open("artifacts/features/weather.json")) if c.startswith("RH")] + ["Afla"]
precip_cols = [c for c in json.load(open("artifacts/features/weather.json")) if c.startswith("PRECTOT")]   + ["Afla"]
ari_cols = [c for c in df3.columns if c.startswith("ARI_")]
# ---------------- MAIN FIGURE ----------------
def create_introstats_figure(df, feature_dict):
    fig, axes = plt.subplots(3,4, figsize=(22,18))
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

# Row 3 (NEW 🔥)
    plot_correlations(axes[8], df, soil_cols, "Soil Correlation")
    plot_correlations(axes[9], df, temp_cols, "Temperature Correlation")
    plot_correlations(axes[10], df, rh_cols, "Humidity (RH2M) Correlation")
    plot_correlations(axes[11], df, precip_cols, "Precipitation Correlation")
    

    
    

    fig.suptitle("Figure 1: Data, Environment & Risk Structure", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    
   

    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/figure1_introstats.png", dpi=300)
    plt.close()

# ---------------- RUN ----------------
if __name__ == "__main__":
    from src.preprocessing import preprocess_data
    dff, feature_dict = preprocess_data("data/preprocessed/2024pg.csv")
    df = pd.read_csv("data/preprocessed/2024pg.csv")
    
    create_introstats_figure(df, feature_dict)