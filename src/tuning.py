from sklearn.model_selection import GridSearchCV

def tune(model, params, X, y):
    grid = GridSearchCV(model, params, cv=3)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_