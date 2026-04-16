import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    precision_score, recall_score, average_precision_score
)
from sklearn.svm import SVC

def apply_safe_smote(X, y):
    from collections import Counter

    class_counts = Counter(y)
    min_class = min(class_counts.values())

    # 🚨 If too few samples → skip SMOTE
    if min_class < 3:
        print("⚠️ Too few minority samples → skipping SMOTE")
        return X, y

    # 🔥 Adaptive k_neighbors
    k_neighbors = min(10, min_class - 1)

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)

    try:
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res
    except Exception as e:
        print(f"⚠️ SMOTE failed: {e}")
        return X, y
    
# ---------------- LOAD DATA ----------------
#df = pd.read_csv("data/selected/2024pg_rf_with_ari.csv")
df3 = pd.read_csv("data/selected/2024pg_rf_with_ari.csv")
df2 = pd.read_csv("data/preprocessed/2024pg.csv")
df = pd.read_csv("data/selected/2024pg_l1.csv")

ari_cols = [c for c in df3.columns if c.startswith("ARI_")]
df3 = df3[ari_cols]

#concat df3 to df but only the columns that are not in df
#df = pd.concat([df, df3.drop(columns=df.columns) if all(c in df3.columns for c in df.columns) else df3], axis=1)
df = pd.concat([df, df3], axis=1)
trg = ["Aflac", "Fumc", "Afla", "Fum"]
for c in trg:
    if c in df.columns:
        df.drop(columns=[c], inplace=True)
#df.drop(columns=trg, inplace=True) if all(c in df.columns for c in trg) else None
print(df.columns)
# ---------------- TARGETS ----------------
df["Aflac"] = (df2["Afla"] > 2).astype(int)
df["Fumc"] = (df2["Fum"] > 4000).astype(int)
df['Afla_raw'] = df2['Afla']
df['Fum_raw'] = df2['Fum']

# ---------------- META ----------------
df["Country"] = df2["Country"].str.lower().str.strip()
df["Crop"] = df2["Crop"].str.lower().str.strip()

# ---------------- CONFIG ----------------
targets = ["Aflac", "Fumc"]
crops = ["maize", "sorghum", "millet"]
countries = ["nigeria", "benin", "cotedvore", "southafrica", "kenya"]

models = {
    "rf": RandomForestClassifier(random_state=42),
    "gb": GradientBoostingClassifier(random_state=42),
    "svm": SVC(random_state=42, probability=True, kernel='rbf')
}

param_grids = {
    "rf": {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_leaf': [1, 2]
    },
    "gb": {
        'n_estimators': [100, 50],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    },
    "svm": {
        'C': [0.1, 1, 10],
        'degree': [3, 5],
        'gamma': ['scale', 'auto']
    }

}

results = []

# ---------------- LOOP ----------------
for crop in crops:

    df_crop = df[df["Crop"] == crop]

    if df_crop.empty:
        print(f"⚠️ No data for crop: {crop}")
        continue

    for target in targets:

        print(f"\n🎯 Target: {target} | 🌾 Crop: {crop}")

        for country in countries:

            print(f"🌍 Test Country: {country}")

            df_test = df_crop[df_crop["Country"] == country]
            df_train = df_crop[df_crop["Country"] != country]

            # 🚨 Skip if empty
            if df_test.empty or df_train.empty:
                print("⚠️ Skipping (empty train/test)")
                continue

            # ---------------- FEATURES ----------------
            if target == 'Aflac':
                X_train = df_train.drop(columns=['Aflac', "Country", "Crop", "Afla_raw"])
            else:
                X_train = df_train.drop(columns=['Fumc', "Country", "Crop", "Fum_raw"])
            y_train = df_train[target]

            if target == 'Aflac':
                X_test = df_test.drop(columns=['Aflac', "Country", "Crop", "Afla_raw"])
            else:
                X_test = df_test.drop(columns=['Fumc', "Country", "Crop", "Fum_raw"])
            y_test = df_test[target]

            # 🚨 Skip if no class diversity
            if len(X_train)<20 or len(X_test)<2:
                print("⚠️ Skipping (too few samples)")
                continue
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                print("⚠️ Skipping (only one class)")
                continue

            # ---------------- SMOTE ----------------
            
            X_train, y_train = apply_safe_smote(X_train, y_train)

            for model_name, base_model in models.items():

                #X_train, y_train = apply_safe_smote(X_train, y_train) if model_name != "rf" else (X_train, y_train)

                print(f"   🤖 Model: {model_name}")

                grid = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grids[model_name],
                    cv=3,
                    scoring="f1",
                    n_jobs=-1
                )

                try:
                    grid.fit(X_train, y_train)
                    model = grid.best_estimator_

                    y_pred = model.predict(X_test)

                    #importances = model.feature_importances_
                    #feature_names = X_train.columns
                    #feature_importance_df = pd.DataFrame({'feature': feature_names,'importance': importances}).sort_values(by='importance', ascending=False)
                    #feature_importance_df = feature_importance_df.head(10)

                    # ---------------- METRICS ----------------
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)

                    try:
                        roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                    except:
                        roc = np.nan

                    ap = average_precision_score(y_test, y_pred)

                    results.append({
                        "target": target,
                        "crop": crop,
                        "test_country": country,
                        "model": model_name,
                        "accuracy": acc,
                        "f1": f1,
                        "precision": prec,
                        "recall": rec,
                        "roc_auc": roc,
                        "avg_precision": ap,
                        "n_train": len(y_train),
                        "n_test": len(y_test),
                        #"feature_importance": feature_importance_df.to_dict(orient='list'),
                        **model.get_params()
                    })

                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue

# ---------------- SAVE RESULTS ----------------
results_df = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)
results_df.to_csv("results/test_results.csv", index=False)

print("\n✅ Results saved to results/test_results.csv")
print(results_df.shape)