from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.base import is_classifier

import numpy as np
import os
import joblib
import shap
import pandas as pd


def train_and_evaluate(X, y, features, model, config,
                       fs_name, model_name, target_name):

    # ---------------- SELECT FEATURES ----------------
    X_subset = X[features]

    # ---------------- SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y,
        test_size=config["train"]["test_size"],
        random_state=config["train"]["random_state"],
        stratify=y
    )

    print(f"\n BEFORE SMOTE: {np.bincount(y_train)}")
    print(f"Classes before SMOTE: {np.unique(y_train)}")

    # ---------------- PIPELINE ----------------
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=config["train"]["random_state"])),
        ("model", model)
    ])

    # ---------------- TRAIN ----------------
    pipeline.fit(X_train, y_train)

    # ---------------- AFTER SMOTE ----------------
    X_res, y_res = pipeline.named_steps['smote'].fit_resample(X_train, y_train)
    print(f" AFTER SMOTE: {np.bincount(y_res)}")
    print(f"Classes after SMOTE: {np.unique(y_res)}")

    # ---------------- SHAP / FEATURE IMPORTANCE ----------------
    shap_file = None
    try:
        model_step = pipeline.named_steps['model']

        # Prepare scaled X_test for SHAP
        X_test_transformed = pipeline.named_steps['scaler'].transform(X_test)
        X_train_transformed = pipeline.named_steps['scaler'].transform(X_train)

        # -------- RANDOM FOREST: use built-in feature_importances_ --------
        if "RandomForest" in str(type(model_step)) or hasattr(model_step, "feature_importances_"):
            mean_importance = model_step.feature_importances_
            shap_df = pd.DataFrame({
                "feature": X_test.columns,
                "shap_value": mean_importance
            }).sort_values(by="shap_value", ascending=False).head(10)
            os.makedirs("artifacts/shap", exist_ok=True)
            shap_file = f"artifacts/shap/{fs_name}_{target_name}_{model_name}_top10.csv"
            shap_df.to_csv(shap_file, index=False)
            print(f" RF top 10 feature importance saved → {shap_file}")

        # -------- Linear models (Lasso, Ridge) --------
        elif "Lasso" in str(type(model_step)) or "Ridge" in str(type(model_step)):
            explainer = shap.LinearExplainer(model_step, X_train_transformed)
            shap_values = explainer.shap_values(X_test_transformed)

        # -------- Neural Network / MLP --------
        elif is_classifier(model_step) and "MLP" in str(type(model_step)):
            explainer = shap.KernelExplainer(
                lambda x: model_step.predict_proba(x)[:, 1],
                shap.kmeans(X_train_transformed, 50)
            )
            shap_values = explainer.shap_values(X_test_transformed)

        # -------- Fallback for other models --------
        else:
            if is_classifier(model_step):
                explainer = shap.KernelExplainer(model_step.predict_proba, shap.kmeans(X_train_transformed, 10))
                shap_values = explainer.shap_values(X_test_transformed)
            else:
                explainer = shap.KernelExplainer(model_step.predict, shap.kmeans(X_train_transformed, 10))
                shap_values = explainer.shap_values(X_test_transformed)

        # Flatten SHAP array safely and save top 10
        if 'shap_values' in locals():
            shap_values_to_use = shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values
            shap_values_to_use = np.atleast_2d(shap_values_to_use)
            mean_shap = np.abs(shap_values_to_use).mean(axis=0)
            shap_df = pd.DataFrame({
                "feature": X_test.columns,
                "shap_value": mean_shap
            }).sort_values(by="shap_value", ascending=False).head(10)
            shap_file = f"artifacts/shap/{fs_name}_{target_name}_{model_name}_top10.csv"
            shap_df.to_csv(shap_file, index=False)
            print(f" SHAP top 10 saved → {shap_file}")

    except Exception as e:
        print("⚠️ SHAP explanation failed:", e)
        shap_file = None

    # ---------------- PREDICT ----------------
    y_pred = pipeline.predict(X_test)
    y_test = y_test.astype(int)

    if not np.issubdtype(y_pred.dtype, np.integer):
        y_pred = (y_pred >= 0.5).astype(int)

    # ---------------- METRICS ----------------
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "r2": r2_score(y_test, y_pred)
    }

    # ---------------- MODEL ID ----------------
    fs_map = {
        "weather": "wo",
        "weather_soil": "ws",
        "weather_soil_agro": "all"
    }

    target_map = {"Aflac": "af", "Fumc": "fu"}

    model_code = model_name[:2]

    model_id = f"{fs_map[fs_name]}{target_map[target_name]}{model_code}"

    # ---------------- SAVE MODEL ----------------
    os.makedirs("artifacts/models", exist_ok=True)
    model_path = f"artifacts/models/{model_id}.pkl"
    joblib.dump(pipeline, model_path)
    print(f" Saved → {model_path}")

    return metrics, pipeline, model_id, shap_file