import mlflow

def start_experiment():
    mlflow.set_experiment("mycotoxin_models")

def log_run(model_name, feature_set, target, metrics, params):
    with mlflow.start_run():
        mlflow.log_param("model", model_name)
        mlflow.log_param("feature_set", feature_set)
        mlflow.log_param("target", target)

        for k, v in params.items():
            mlflow.log_param(k, v)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)