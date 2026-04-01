import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import os

# ---------------- FORWARD SELECTION ----------------
def forward_selection(input_csv="data/processed/2024pg.csv", target_col="Aflac", model=None, cv=5):
    """
    Forward feature selection using cross-validated F1 score.
    Returns selected dataframe and list of selected features.
    Saves CSV with suffix '_for.csv'.
    """
    df = pd.read_csv(input_csv)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if model is None:
        model = RandomForestClassifier(random_state=42)

    remaining_features = list(X.columns)
    selected_features = []
    best_score = 0

    while remaining_features:
        scores = {}
        for feature in remaining_features:
            current_features = selected_features + [feature]
            score = cross_val_score(clone(model), X[current_features], y, cv=cv, scoring='f1').mean()
            scores[feature] = score
        best_feature = max(scores, key=scores.get)
        if scores[best_feature] > best_score:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            best_score = scores[best_feature]
        else:
            break  # Stop if adding features does not improve

    print(f"Selected features (forward): {selected_features}")
    df_selected = df[selected_features + [target_col]]

    out_file = input_csv.replace(".csv", "_for.csv")
    os.makedirs("data/selected", exist_ok=True)
    df_selected.to_csv(os.path.join("data/selected", os.path.basename(out_file)), index=False)
    print(f"✅ Forward selection CSV saved → data/selected/{os.path.basename(out_file)}")
    return df_selected, selected_features


# ---------------- BACKWARD SELECTION ----------------
def backward_selection(input_csv="data/processed/2024pg.csv", target_col="Aflac", model=None, cv=5):
    """
    Backward feature selection using cross-validated F1 score.
    Returns selected dataframe and list of selected features.
    Saves CSV with suffix '_back.csv'.
    """
    df = pd.read_csv(input_csv)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if model is None:
        model = RandomForestClassifier(random_state=42)

    selected_features = list(X.columns)
    best_score = cross_val_score(clone(model), X[selected_features], y, cv=cv, scoring='f1').mean()

    while len(selected_features) > 1:
        scores = {}
        for feature in selected_features:
            current_features = [f for f in selected_features if f != feature]
            score = cross_val_score(clone(model), X[current_features], y, cv=cv, scoring='f1').mean()
            scores[feature] = score
        worst_feature = min(scores, key=scores.get)
        if scores[worst_feature] >= best_score:
            selected_features.remove(worst_feature)
            best_score = scores[worst_feature]
        else:
            break  # Stop if removing features reduces score

    print(f"Selected features (backward): {selected_features}")
    df_selected = df[selected_features + [target_col]]

    out_file = input_csv.replace(".csv", "_back.csv")
    os.makedirs("data/selected", exist_ok=True)
    df_selected.to_csv(os.path.join("data/selected", os.path.basename(out_file)), index=False)
    print(f"✅ Backward selection CSV saved → data/selected/{os.path.basename(out_file)}")
    return df_selected, selected_features


# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Example usage: call any function you want
    forward_selection()
    backward_selection()