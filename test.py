import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, average_precision_score
# import sensistivity_score, specificity_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


df2 = pd.read_csv("data/preprocessed/2024pw_weekly.csv")
df2.drop(columns=["Latitude", "Longitude", "Sowdate", "Harvestdate"], inplace=True)
# clip teh columns on week 20
week_cols = [c for c in df2.columns if c.startswith(("T2M", "PRECTOT", "RH2M", "ALLSKY_SFC_SW_DWN", "GWETROOT", "EVPTRNS", "WS2M"))]
week20_cols = [c for c in week_cols if c.endswith("w20")]
week20_cols_upwards_till_w40 = [c for c in week_cols if any(c.endswith(f"w{w}") for w in range(0, 21))]
df2 = df2[week20_cols_upwards_till_w40]
#df2 to ontain values not in week20_cols_upwards_till_w40

# fill missing values with median
df2.fillna(df2.median(), inplace=True)
# 

df = pd.read_csv("data/selected/2024pg_l1.csv")
df.drop(columns=[c for c in df.columns if c in ["Latitude", "Longitude", "Sowdate", "Harvestdate", "Aflac", "Fumc", "Afla", "Fum"]], inplace=True)
df4 = pd.read_csv("data/ari_values.csv")
df4.drop(columns=[c for c in df4.columns if c in ["Latitude", "Longitude", "Sowdate", "Harvestdate", "Aflac", "Fumc", "Afla", "Fum"]], inplace=True)
df3 = pd.read_csv("data/preprocessed/2024pg.csv")
df2['Aflac'] = (df3['Afla'] > 4).astype(int)
df2['Fumc'] = (df3['Fum'] > 4000).astype(int)
df = pd.concat([df4 ,df, df2], axis=1)
print(f"Final columns: {df.columns}, Total: {df.shape}")


df = pd.read_csv("data/preprocessed/2024pw_daily8p120.csv")
df.drop(columns=["Latitude", "Longitude", "Sowdate"], inplace=True)
df['Aflac'] = (df3['Afla'] > 4).astype(int)
df['Fumc'] = (df3['Fum'] > 4000).astype(int) 

df = df2.copy()

print(f"Final columns: {df.columns}, Total: {df.shape}")
mod = XGBClassifier(random_state=42)
param_grid = {
    'n_estimators': [400],
    'max_depth': [20, 15],
    'learning_rate': [0.1],
    #'subsample': [0.8],
    'scale_pos_weight': [len(df[df['Aflac'] == 0]) / len(df[df['Aflac'] == 1])]

}
model = GridSearchCV(estimator=mod, param_grid=param_grid, cv=5, scoring='f1')


X = df.drop(columns=['Aflac'])
#X = (df['ARI_total'].values.reshape(-1, 1)).astype(float)  # Reshape for sklearn
y = df["Aflac"]  # or "Fumc"

# train test split
#from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=36)
print("Before SMOTEenn, training set class distribution:", y_train.value_counts())
print("Before SMOTEenn, test set class distribution:", y_test.value_counts())
# if column name its starts with Country_be
#X_test = X[X["Crop_Maize"] == 1]
#y_test = y[X["Crop_Maize"] == 1]
#X_train = X[X["Crop_Maize"] == 0]
#y_train = y[X["Crop_Maize"] == 0]


# smote enn to balance the dataset
smotenn = SMOTEENN(random_state=42, smote=SMOTE(k_neighbors=3, sampling_strategy=0.6),
                enn=EditedNearestNeighbours(n_neighbors=3)
                )
X_train, y_train = smotenn.fit_resample(X_train, y_train)

#smote = SMOTE(random_state=42)
#_train, y_train = smote.fit_resample(X_train, y_train)
print("After SMOTEenn, training set class distribution:", y_train.value_counts())

scaler = StandardScaler()
scaler = scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#print(X_train.columns)
model.fit(X_train, y_train)

model = model.best_estimator_
model = CalibratedClassifierCV(model, method='sigmoid')
model.fit(X_train, y_train)
print("Best Hyperparameters:", model.get_params())
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.6).astype(int)
print(classification_report(y_test, y_pred))
print("-" * 50)
print(confusion_matrix(y_test, y_pred, labels=model.classes_))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
#print("Sensitivity (Recall):", sensitivity_score(y_test, y_pred))
#print("Specificity:", specificity_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Average Precision Score:", average_precision_score(y_test, y_pred))
# amke a confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("TP FP FN TN:", tp, fp, fn, tn)
#print("R2 Score:", r2_score(y_test, y_pred))
#print("MAE:", mean_absolute_error(y_test, y_pred))
#print("MSE:", mean_squared_error(y_test, y_pred))

#plot var imp
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)
data=feature_importance_df.head(100)["feature"].tolist()
print(data[:20])
data.to_json("features/feature_daily8p120_importance.json", orient="records")
