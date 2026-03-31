from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
import numpy as np

# Helper to threshold predictions
def threshold_predictions(y_pred):
    if not np.issubdtype(y_pred.dtype, np.integer):
        y_pred = (y_pred >= 0.5).astype(int)
    return y_pred

# Custom scorers
def thresholded_accuracy(y_true, y_pred):
    y_pred = threshold_predictions(y_pred)
    return accuracy_score(y_true, y_pred)

def thresholded_precision(y_true, y_pred):
    y_pred = threshold_predictions(y_pred)
    return precision_score(y_true, y_pred)

def thresholded_recall(y_true, y_pred):
    y_pred = threshold_predictions(y_pred)
    return recall_score(y_true, y_pred)

def thresholded_f1(y_true, y_pred):
    y_pred = threshold_predictions(y_pred)
    return f1_score(y_true, y_pred)

# Main cross-validation function
def run_cv(model, X, y, cv=5):
    # Ensure y is binary integers
    y = y.astype(int)

    # Standardize features
    X = StandardScaler().fit_transform(X)

    # Define scoring dictionary
    scoring = {
        "accuracy": make_scorer(thresholded_accuracy),
        "precision": make_scorer(thresholded_precision),
        "recall": make_scorer(thresholded_recall),
        "f1": make_scorer(thresholded_f1)
    }

    # Run cross-validation
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)

    # Print mean scores
    print({k: v.mean() for k, v in scores.items()})

    # Return as dictionary
    return {m: scores[f"test_{m}"].mean() for m in scoring}