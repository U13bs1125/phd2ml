from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np

def train_and_evaluate(X, y, features, model, config):
    # Select only the given features
    X_subset = X[features]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y,
        test_size=config["train"]["test_size"],
        random_state=config["train"]["random_state"],
        stratify=y  # keep class distribution in test set
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Apply SMOTE only on the training set
    smote = SMOTE(random_state=config["train"]["random_state"])
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Convert continuous outputs to binary if needed
    if not np.issubdtype(y_pred.dtype, np.integer):
        y_pred = (y_pred >= 0.5).astype(int)

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    return metrics, model