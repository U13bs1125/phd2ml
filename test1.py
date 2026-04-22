import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_auc_score, precision_score, recall_score,
    average_precision_score, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.calibration import CalibratedClassifierCV

# ==============================
# 1. LOAD CLEAN DATA
# ==============================
df = pd.read_csv("data/processed/engineered_features.csv")
df2 = pd.read_csv("data/preprocessed/2024pg.csv")
df3 = pd.read_csv("data/ari_values.csv")
# Create labels
df['Aflac'] = (df2['Afla'] > 4).astype(int)
df['Fumc'] = (df2['Fum'] > 4000).astype(int)
df = pd.concat([df, df3], axis=1)
# Drop raw targets if present
df = df.drop(columns=[col for col in ['Afla', 'Fum'] if col in df.columns])

print("Final shape:", df.shape)

# ==============================
# 2. DEFINE FEATURES / TARGET
# ==============================
X = df.drop(columns=['Aflac'])
y = df['Aflac']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("Train distribution:\n", y_train.value_counts())
print("Test distribution:\n", y_test.value_counts())

# ==============================
# 3. SCALE FEATURES
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Class imbalance weight
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# ==============================
# 4. DEFINE MODELS + PARAM GRIDS
# ==============================
models = {

    "RandomForest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [300, 500],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        }
    ),

    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "n_estimators": [200, 300],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]
        }
    ),

    "SVM": (
        SVC(probability=True, class_weight='balanced', random_state=42),
        {
            "C": [0.5, 1, 5],
            "gamma": ["scale", 0.01],
            "kernel": ["rbf"]
        }
    ),

    "XGBoost": (
        XGBClassifier(random_state=42, eval_metric='logloss'),
        {
            "n_estimators": [300, 500],
            "max_depth": [6, 10],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "scale_pos_weight": [scale_pos_weight]
        }
    )
}

# ==============================
# 5. TRAIN + EVALUATE
# ==============================
results = {}

for name, (model, params) in models.items():

    print(f"\n🚀 Training {name}...")

    grid = GridSearchCV(
        model,
        param_grid=params,
        cv=3,
        scoring='f1',
        n_jobs=-1
    )

    # Use scaled data for all models (safe choice)
    grid.fit(X_train_scaled, y_train)

    best_model = grid.best_estimator_

    # Calibration
    calibrated = CalibratedClassifierCV(best_model, cv=3, method='sigmoid')
    calibrated.fit(X_train_scaled, y_train)

    # Probabilities
    y_prob = calibrated.predict_proba(X_test_scaled)[:, 1]

    # ==============================
    # 6. OPTIMAL THRESHOLD
    # ==============================
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

    best_threshold = thresholds[np.argmax(recall)]
    y_pred = (y_prob >= best_threshold).astype(int)

    # ==============================
    # 7. METRICS
    # ==============================
    print("Best Params:", grid.best_params_)
    print("Best Threshold:", best_threshold)

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    precision_score_val = precision_score(y_test, y_pred)
    recall_score_val = recall_score(y_test, y_pred)

    print("Accuracy:", acc)
    print("F1:", f1)
    print("ROC AUC:", roc)
    print("Precision:", precision_score_val)
    print("Recall:", recall_score_val)


    results[name] = f1

    # ==============================
    # 8. FEATURE IMPORTANCE (if exists)
    # ==============================
    try:
        base_model = calibrated.calibrated_classifiers_[0].estimator

        if hasattr(base_model, "feature_importances_"):
            importances = base_model.feature_importances_

            feat_imp = pd.DataFrame({
                "feature": X.columns,
                "importance": importances
            }).sort_values(by="importance", ascending=False)

            print("\nTop Features:")
            print(feat_imp.head(15))

        else:
            print("No feature importance for this model.")

    except Exception as e:
        print("Feature importance error:", e)

# ==============================
# 9. BEST MODEL SUMMARY
# ==============================
print("\n🏆 Final Model Comparison:")
for k, v in results.items():
    print(f"{k}: F1 = {v:.4f}")