from src.config_loader import load_config
from src.models import get_models
from src.tuning import tune
from figure2_performance import create_figure2_performance
from src.mlflow_tracker import start_experiment, log_run
from src.preprocessing import preprocess_data
from src.features import get_feature_sets
from src.trainnn import train_and_evaluate
from figure3_featureimp import plot_shap_grid
from src.ft_selection import shap_selection
from figure4_generalization import create_generalization_figure
from src.data import load_data
from figure5_climatechange import create_climate_figure
from figure1_introstats import create_introstats_figure
import matplotlib.pyplot as plt

import pandas as pd


def main():

    #forward_selection()

    # ---------------- PREPROCESS ----------------
    dfg, feature_dict = preprocess_data("data/preprocessed/2024pg.csv")

    config = load_config()
    targets = config["data"]["targets"]

    feature_sets = get_feature_sets(feature_dict)
    models = get_models(config)

    all_results = []

    start_experiment()

    # ---------------- LOOP ----------------
    for target in targets:

        print(f"\n Target: {target}")

        y = dfg[target].astype(int)

        #  FIX: drop ALL targets  
        X = dfg.drop(columns=targets)

        for fs_name, features in feature_sets.items():

            features = [f for f in features if f in X.columns]

            for model_name, model in models.items():

                # tuning
                if model_name in config["tuning"]:
                    model, best_params = tune(
                        model,
                        config["tuning"][model_name],
                        X[features],
                        y
                    )
                else:
                    best_params = {}

                #  FIX: pass names
                metrics, trained_model, model_id, shap_file = train_and_evaluate(
                    X, y, features, model, config,
                    fs_name, model_name, target
                )

                result_row = {
                    "model_id": model_id,
                    "target": target,
                    "feature_set": fs_name,
                    "model": model_name,
                    **metrics,
                    **best_params
                }

                all_results.append(result_row)

    df = pd.DataFrame(all_results)
    import os
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/metrics.csv", index=False)

    print("Results saved:", df.shape)

    create_figure2_performance(df)
    plot_shap_grid()
    create_generalization_figure(df = load_data())
    create_climate_figure()


if __name__ == "__main__":
    main()


# by the side
from src.preprocessing import preprocess_data
df, feature_dict = preprocess_data("data/preprocessed/2024pg.csv")
dff = pd.read_csv("data/preprocessed/2024pg.csv")
create_introstats_figure(dff, feature_dict)