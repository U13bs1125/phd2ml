import shap

def compute_shap(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    return shap_values

import matplotlib.pyplot as plt
import os

def plot_shap(shap_values, X, name):
    shap.summary_plot(shap_values, X, show=False)

    os.makedirs("results/plots/shap", exist_ok=True)
    plt.savefig(f"results/plots/shap/{name}.png")
    plt.close()