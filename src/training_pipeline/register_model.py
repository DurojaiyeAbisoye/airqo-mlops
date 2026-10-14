import warnings
from pathlib import Path

import feast.mlflow
from feast import FeatureStore
from mlflow.entities.logged_model import LoggedModel

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_REPO = REPO_ROOT / "src/feature_repo"
FEATURE_SERVICE_NAME = "pm25_regression"
REGISTERED_MODEL_NAME = f"{FEATURE_SERVICE_NAME} model"
CHAMPION_ALIAS = "champion"
LOGGED_MODEL_FILTER = "name = 'pm25_regression'"


def create_feature_store() -> FeatureStore:
    store = FeatureStore(
        repo_path=str(FEATURE_REPO),
        fs_yaml_file=FEATURE_REPO / "feature_store.yaml",
    )
    feast.mlflow.init(store=store)
    return store


def find_best_logged_model(store: FeatureStore) -> LoggedModel:
    mlflow_client = feast.mlflow.MlflowClient(store.mlflow._tracking_uri)
    experiment_info = feast.mlflow.get_experiment_by_name(store.project)
    if experiment_info is None:
        raise RuntimeError(
            f"MLflow experiment {store.project!r} does not exist."
        )

    logged_models = mlflow_client.search_logged_models(
        experiment_ids=[experiment_info.experiment_id],
        max_results=1,
        order_by=[{"field_name": "metrics.val_rmse", "ascending": True}],
        filter_string=LOGGED_MODEL_FILTER,
    )
    if not logged_models:
        raise RuntimeError(
            "No logged models with validation RMSE were found for "
            f"filter {LOGGED_MODEL_FILTER!r}."
        )
    return logged_models[0]


def register_best_model(store: FeatureStore) -> None:
    logged_model = find_best_logged_model(store)
    print(logged_model)

    model_version = store.mlflow.register_model(
        logged_model.model_uri,
        name=REGISTERED_MODEL_NAME,
    )
    print("Model Registered!")

    mlflow_client = feast.mlflow.MlflowClient(store.mlflow._tracking_uri)
    mlflow_client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version=model_version.version,
    )
    print("model Aliased!")


def main() -> None:
    warnings.filterwarnings("ignore")
    store = create_feature_store()
    register_best_model(store)


if __name__ == "__main__":
    main()
