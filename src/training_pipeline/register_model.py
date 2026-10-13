# register best model to mlflow model registry
import warnings
import feast.mlflow
from feast import FeatureStore
from pathlib import Path

feature_service_name = "pm25_regression"

warnings.filterwarnings("ignore")
# initialize the feature store
REPO_PATH = Path(__file__).resolve().parents[2]

store = FeatureStore(
    repo_path=f"{REPO_PATH}/src/feature_repo/",
    fs_yaml_file=Path(f"{REPO_PATH}/src/feature_repo/feature_store.yaml"),
)
feast.mlflow.init(store=store)

# init client
mlflow_client = feast.mlflow.MlflowClient(store.mlflow._tracking_uri)

# search experiments
experiment_name = store.project
experiment_info = feast.mlflow.get_experiment_by_name(experiment_name)

logged_model_info = mlflow_client.search_logged_models(
    experiment_ids=[experiment_info.experiment_id],
    max_results=1,
    order_by=[{"field_name": "metrics.val_rmse", "ascending": True}],
    filter_string="name = 'pm25_regression'",
)[0]
print(logged_model_info)
mv = store.mlflow.register_model(
    logged_model_info.model_uri, name=f"{feature_service_name} model"
)
print("Model Registered!")
mlflow_client.set_registered_model_alias(
    name=f"{feature_service_name} model", alias="champion", version=mv.version
)
print("model Aliased!")
