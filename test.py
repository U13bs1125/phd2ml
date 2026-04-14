import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, average_precision_score
# import sensistivity_score, specificity_score


df = pd.read_csv("data/selected/2024pg_rf_with_ari.csv") 
df.drop(columns=['Aflac'], inplace=True)
df2 = pd.read_csv("data/preprocessed/2024pg.csv")
df['Aflac'] = (df2['Afla'] > 4).astype(int)
df['Country'] = df2['Country']
df['Crop'] = df2['Crop']
print(df['Crop'].value_counts())
#print(df['Country'].value_counts())
df = df[df['Crop'] == 'maize'].drop(columns=['Crop'])
print(df['Country'].value_counts())
df3 = df[df['Country'] == 'kenya']
df = df[df['Country'] != 'kenya'].drop(columns=['Country'])


mod = RandomForestClassifier(random_state=42, class_weight='balanced')
param_grid = {
    'n_estimators': [300, 400, 50],
    'max_depth': [50, 10, 20, 5],
    'min_samples_split': [2],
    'min_samples_leaf': [1, 2],
   

}
model = GridSearchCV(estimator=mod, param_grid=param_grid, cv=3, scoring='f1')

ari_cols = [c for c in df.columns if c.startswith("ARI_")]
#X = df[ari_cols]
X = df.drop(columns=['Aflac'])
#X = (df['ARI_total'].values.reshape(-1, 1)).astype(float)  # Reshape for sklearn
y = df["Aflac"]  # or "Fumc"

# train test split
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train = X, y
X_test, y_test = df3.drop(columns=['Aflac', 'Country']), df3['Aflac'] 
print("Training set class distribution:", y_train.value_counts())
print("Test set class distribution:", y_test.value_counts())

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("After SMOTE, training set class distribution:", y_train.value_counts())


#scaler = StandardScaler()
#scaler = scaler.fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)
#print(X_train.columns)
model.fit(X_train, y_train)

model = model.best_estimator_
print("Best Hyperparameters:", model.get_params())
y_pred = model.predict(X_test)
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
