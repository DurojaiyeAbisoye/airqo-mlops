from pathlib import Path

import feast.mlflow
from feast import FeatureStore

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_SERVICE_NAME = "pm25_regression"
FEATURE_REPO = REPO_ROOT / "src/feature_repo"


def create_feature_store() -> FeatureStore:
    store = FeatureStore(
        repo_path=str(FEATURE_REPO),
        fs_yaml_file=FEATURE_REPO / "feature_store.yaml",
    )
    feast.mlflow.init(store=store)
    feast.mlflow.autolog()
    return store
