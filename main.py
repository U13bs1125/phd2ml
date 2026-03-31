from src.config_loader import load_config
from src.models import get_models
from src.tuning import tune
from src.crossval import run_cv
from src.visualize import create_master_figure
from src.mlflow_tracker import start_experiment, log_run
from src.preprocessing import preprocess_data
from src.features import get_feature_sets

import pandas as pd


def main():

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

        print(f"\n🎯 Target: {target}")

        y = dfg[target]
        X = dfg.drop(columns=targets)

        for fs_name, features in feature_sets.items():

            print(f"Feature set: {fs_name} ({len(features)} features)")

            # Ensure only valid columns
            features = [f for f in features if f in X.columns]

            X_sub = X[features]

            for model_name, model in models.items():

                print(f"Model: {model_name}")

                # ---------------- TUNING ----------------
                if model_name in config["tuning"]:
                    model, params = tune(
                        model,
                        config["tuning"][model_name],
                        X_sub,
                        y
                    )
                else:
                    params = {}

                # ---------------- CV ----------------
                metrics = run_cv(model, X_sub, y)

                # ---------------- LOG ----------------
                log_run(model_name, fs_name, target, metrics, params)

                all_results.append({
                    "target": target,
                    "feature_set": fs_name,
                    "model": model_name,
                    **metrics
                })

    # ---------------- SAVE ----------------
    df = pd.DataFrame(all_results)
    import os
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/metrics.csv", index=False)

    # ---------------- FIGURE ----------------
    create_master_figure(df)


if __name__ == "__main__":
    main()