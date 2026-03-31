from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def get_models(config):
    return {
        "rf": RandomForestClassifier(),
        "xgb": XGBClassifier(),
        "lasso": Lasso(),
        "nn": MLPClassifier()
    }