import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
accuracy_score, f1_score, precision_score, recall_score,
roc_auc_score, average_precision_score, confusion_matrix
)

# =========================

# LOAD DATA

# =========================

df = pd.read_csv("data/processed/7params_weekly.csv")

df = df.dropna()

# =========================

# FIX ARI (complex string -> float)

# =========================

ari = df["ARI_post"].apply(lambda x: complex(x).real).astype(np.float32)

# =========================

# FEATURES / TARGET

# =========================

X = df.drop(columns=[c for c in df.columns if c in ['Aflac', 'Afla', 'Fum', 'ARI_post']])
y = df["Aflac"].astype(np.float32)

# Convert ALL features to numeric

X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

# =========================

# SPLIT

# =========================

X_train, X_test, y_train, y_test, ari_train, ari_test = train_test_split(
X.values, y.values, ari.values, test_size=0.2, random_state=42
)

# =========================

# TORCH

# =========================

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

ari_train = torch.tensor(ari_train, dtype=torch.float32).view(-1, 1)

# =========================

# PINN MODEL

# =========================

class PINN(nn.Module):
def __init__(self, input_dim):
super().__init__()
self.net = nn.Sequential(
nn.Linear(input_dim, 64),
nn.ReLU(),
nn.Linear(64, 32),
nn.ReLU(),
nn.Linear(32, 1),
nn.Sigmoid()
)

def forward(self, x):
    return self.net(x)

```
def forward(self, x):
    return self.net(x)
```

model = PINN(X_train.shape[1])

optimizer = optim.Adam(model.parameters(), lr=0.001)
bce = nn.BCELoss()
mse = nn.MSELoss()

lambda_phys = 0.5

# =========================

# TRAIN

# =========================

for epoch in range(100):
    optimizer.zero_grad()
    
    preds = model(X_train)
    
    loss_data = bce(preds, y_train)
    
    # normalize ARI
    ari_norm = ari_train / (ari_train.max() + 1e-6)
    
    loss_phys = mse(preds, ari_norm)
    
    loss = loss_data + lambda_phys * loss_phys
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss {loss.item():.4f}")

```
optimizer.zero_grad()

preds = model(X_train)

loss_data = bce(preds, y_train)

# normalize ARI
ari_norm = ari_train / (ari_train.max() + 1e-6)

loss_phys = mse(preds, ari_norm)

loss = loss_data + lambda_phys * loss_phys

loss.backward()
optimizer.step()

if epoch % 10 == 0:
    print(f"Epoch {epoch} | Loss {loss.item():.4f}")
```

# =========================

# EVALUATION

# =========================

preds = model(X_test).detach().numpy().flatten()
y_true = y_test.numpy().flatten()
y_pred = (preds > 0.5).astype(int)

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
roc = roc_auc_score(y_true, preds)
ap = average_precision_score(y_true, preds)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
spec = tn / (tn + fp)

print("\n=== RESULTS ===")
print("Accuracy:", acc)
print("F1:", f1)
print("Precision:", prec)
print("Recall:", rec)
print("ROC-AUC:", roc)
print("PR-AUC:", ap)
print("Specificity:", spec)
print("TP FP FN TN:", tp, fp, fn, tn)

# =========================

# SHAP (FIXED)

# =========================

# Convert tensors to numpy

X_train_np = X_train.detach().numpy()
X_test_np = X_test.detach().numpy()

# Wrap model for SHAP

def model_fn(x):
x_tensor = torch.tensor(x, dtype=torch.float32)
with torch.no_grad():
return model(x_tensor).numpy()

# Use SHAP explainer

explainer = shap.Explainer(model_fn, X_train_np)
shap_values = explainer(X_test_np)

# Plot

shap.summary_plot(shap_values, X_test_np)
