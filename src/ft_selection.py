import pandas as pd
import numpy as np
import os

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone, is_classifier
from sklearn.preprocessing import StandardScaler

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False


# ---------------- PREPARE DATA ----------------
def prepare_data(input_csv, target_col="Afla"):
    df = pd.read_csv(input_csv)

    # Binary target
    df["Aflac"] = np.where(df[target_col] > 10, 1, 0)

    # Remove leakage
    drop_cols = ["Afla", "Aflac", "Fumc"]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["Aflac"]

    return df, X, y


# ---------------- SAVE FUNCTION ----------------
def save_selected(df, features, method_name, input_csv):
    os.makedirs("data/selected", exist_ok=True)

    df_selected = df[features + ["Aflac"]]

    filename = os.path.basename(input_csv).replace(".csv", f"_{method_name}.csv")
    out_path = os.path.join("data/selected", filename)

    df_selected.to_csv(out_path, index=False)

    print(f"✅ Saved → {out_path}")
    return df_selected


# ---------------- 1. FORWARD SELECTION ----------------
#def forward_selection(input_csv="data/processed/2024pg.csv", min_features=30, cv=3):

    df, X, y = prepare_data(input_csv)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    remaining = list(X.columns)
    selected = []
    best_score = 0

    print("\n🚀 Forward Selection")

    while remaining:
        scores = {}

        for f in remaining:
            feats = selected + [f]

            try:
                score = cross_val_score(
                    clone(model),
                    X[feats],
                    y,
                    cv=cv,
                    scoring="f1",
                    n_jobs=-1
                ).mean()
            except Exception as e:
                continue

            scores[f] = score

        if not scores:
            break

        best_f = max(scores, key=scores.get)

        if scores[best_f] > best_score or len(selected) < min_features:
            selected.append(best_f)
            remaining.remove(best_f)
            best_score = max(best_score, scores[best_f])

            print(f" + {best_f} | score={best_score:.4f}")
        else:
            break

    print(f"\n✅ Forward selected ({len(selected)}) features")
    return save_selected(df, selected, "forward", input_csv), selected


# ---------------- 2. RF IMPORTANCE ----------------
def rf_importance_selection(input_csv="data/processed/2024pg.csv", top_n=50):

    df, X, y = prepare_data(input_csv)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)

    importances = model.feature_importances_

    feat_imp = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    selected = feat_imp.head(top_n)["feature"].tolist()

    print(f"🌲 RF selected ({len(selected)})")
    return save_selected(df, selected, "rf", input_csv), selected


# ---------------- 3. SHAP SELECTION (ROBUST) ----------------
def shap_selection(input_csv="data/processed/2024pg.csv", top_n=50):

    if not SHAP_AVAILABLE:
        print("⚠️ SHAP not available")
        return None, []

    df, X, y = prepare_data(input_csv)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    print("⚡ Computing SHAP / Feature Importance...")

    shap_df = None

    try:
        # -------- RF: use built-in importance (FAST + STABLE) --------
        if hasattr(model, "feature_importances_"):
            mean_importance = model.feature_importances_

            shap_df = pd.DataFrame({
                "feature": X.columns,
                "importance": mean_importance
            }).sort_values(by="importance", ascending=False)

            print("✅ Used RF feature_importances_")

        # -------- Linear fallback (rare case) --------
        elif "Linear" in str(type(model)):
            explainer = shap.LinearExplainer(model, X)
            shap_values = explainer.shap_values(X)

        # -------- General SHAP logic --------
        else:
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                print("✅ Used TreeExplainer")

            except Exception as e1:
                print("⚠️ TreeExplainer failed:", e1)

                try:
                    explainer = shap.Explainer(model, X)
                    shap_values = explainer(X).values
                    print("✅ Used shap.Explainer")

                except Exception as e2:
                    print("⚠️ shap.Explainer failed:", e2)

                    # FINAL fallback
                    try:
                        background = shap.kmeans(X, 30)

                        if is_classifier(model):
                            explainer = shap.KernelExplainer(
                                lambda x: model.predict_proba(x)[:, 1],
                                background
                            )
                        else:
                            explainer = shap.KernelExplainer(
                                model.predict,
                                background
                            )

                        shap_values = explainer.shap_values(X[:100])
                        print("✅ Used KernelExplainer (fallback)")

                    except Exception as e3:
                        print("❌ SHAP failed completely:", e3)
                        return None, []

        # -------- PROCESS SHAP VALUES --------
        if shap_df is None:

            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            shap_values = np.atleast_2d(shap_values)

            mean_shap = np.abs(shap_values).mean(axis=0)

            shap_df = pd.DataFrame({
                "feature": X.columns[:len(mean_shap)],
                "importance": mean_shap
            }).sort_values(by="importance", ascending=False)

        # -------- SELECT TOP --------
        selected = shap_df.head(top_n)["feature"].tolist()

        print(f"⚡ Selected ({len(selected)}) features using SHAP logic")

        return save_selected(df, selected, "shap", input_csv), selected

    except Exception as e:
        print("⚠️ SHAP selection failed:", e)
        return None, []

# ---------------- 4. SELECTFROMMODEL ----------------
def selectfrommodel_selection(input_csv="data/processed/2024pg.csv"):

    df, X, y = prepare_data(input_csv)

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    selector = SelectFromModel(model, threshold="median")
    selector.fit(X, y)

    selected = X.columns[selector.get_support()].tolist()

    print(f"🎯 SelectFromModel selected ({len(selected)})")
    return save_selected(df, selected, "sfm", input_csv), selected


# ---------------- 5. L1 SELECTION ----------------
def l1_selection(input_csv="data/processed/2024pg.csv"):

    df, X, y = prepare_data(input_csv)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        max_iter=3000
    )
    model.fit(X_scaled, y)

    selected = X.columns[model.coef_[0] != 0].tolist()

    print(f"🧮 L1 selected ({len(selected)})")
    return save_selected(df, selected, "l1", input_csv), selected


# ---------------- MAIN ----------------
if __name__ == "__main__":

    input_file = "data/processed/2024pg.csv"

    # Call any method you want
    #forward_selection(input_file)
    rf_importance_selection(input_file)
    shap_selection(input_file)
    selectfrommodel_selection(input_file)
    l1_selection(input_file)