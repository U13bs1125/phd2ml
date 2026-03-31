from src.config_loader import load_config
from src.models import get_models
from src.tuning import tune
from src.train import train_and_evaluate  # <-- use the new train.py function
from src.visualize import create_master_figure
from src.mlflow_tracker import start_experiment, log_run
from src.preprocessing import preprocess_data
from src.features import get_feature_sets

import pandas as pd
import os

def main():

    # ---------------- PREPROCESS ----------------
    dfg, feature_dict = preprocess_data("data/preprocessed/2024pg.csv")

    config = load_config()
    targets = config["data"]["targets"]

    feature_sets = get_feature_sets(feature_dict)
    models = get_models(config)

    all_resultss = []

    start_experiment()

    # ---------------- LOOP ----------------
    for target in targets:

        print(f"\n🎯 Target: {target}")

        y = dfg[target].astype(int)  # ensure binary integer
        X = dfg.drop(columns=targets)

        for fs_name, features in feature_sets.items():

            print(f"Feature set: {fs_name} ({len(features)} features)")

            # Ensure only valid columns
            features = [f for f in features if f in X.columns]

            for model_name, model in models.items():

                print(f"Model: {model_name}")

                # ---------------- TUNING ----------------
                if model_name in config["tuning"]:
                    model, params = tune(
                        model,
                        config["tuning"][model_name],
                        X[features],
                        y
                    )
                else:
                    params = {}

                # ---------------- TRAIN & EVALUATE ----------------
                metrics, trained_model = train_and_evaluate(
                    X, y, features, model, config
                )

                # ---------------- LOG ----------------
                log_run(model_name, fs_name, target, metrics, params)

                all_resultss.append({
                    "target": target,
                    "feature_set": fs_name,
                    "model": model_name,
                    **metrics
                })

    # ---------------- SAVE ----------------
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(all_resultss)
    df.to_csv("results/metrics_nocv.csv", index=False)
    #save the trained models
    for result in all_resultss:
        model_name = result["model"]
        fs_name = result["feature_set"]
        target = result["target"]
        trained_model = result.get("trained_model")
        if trained_model is not None:
            os.makedirs("results/models", exist_ok=True)
            model_path = f"results/models/{model_name}_{fs_name}_{target}.pkl"
            pd.to_pickle(trained_model, model_path)

    # ---------------- FIGURE ----------------
    create_master_figure(df)


if __name__ == "__main__":
    main()