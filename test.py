from narwhals import col
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, roc_auc_score

df = pd.read_csv("data/weather_soil_agro_with_ari.csv")
df['Aflac'] = (df['Afla'] > 2).astype(int)

mod = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [300, 200, 400],
    'max_depth': [50, 10, 20, None],
    'min_samples_split': [2, 5, None],
    'min_samples_leaf': [1, 2],
   

}
model = GridSearchCV(estimator=mod, param_grid=param_grid, cv=3)

ari_cols = [c for c in df.columns if c.startswith("ARI_")]
#X = df[ari_cols]
X = df.drop(columns=['Afla', 'Aflac'])
#X = (df['ARI_total'].values.reshape(-1, 1)).astype(float)  # Reshape for sklearn
y = df["Aflac"]  # or "Fumc"

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set class distribution:", y_train.value_counts())

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("After SMOTE, training set class distribution:", y_train.value_counts())

scaler = StandardScaler()
scaler = scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.columns)
model.fit(X_train, y_train)

model = model.best_estimator_
print("Best Hyperparameters:", model.get_params())
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("-" * 50)
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

#plot var imp
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), palette='viridis')
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()  
plt.show()
