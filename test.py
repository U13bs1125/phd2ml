import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_clustered_corr(df):
    corr = df.corr()

    sns.clustermap(
        corr,
        cmap="coolwarm",
        center=0,
        figsize=(10,10),
        method="ward",
        linewidths=0.5
    )

    plt.title("Clustered Feature Correlation")
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("data\\selected\\2024pg_shap.csv")
    plot_clustered_corr(df.select_dtypes(include=np.number).fillna(0))