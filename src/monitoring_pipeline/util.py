from pathlib import Path

import feast.mlflow
from catboost import CatBoostRegressor
from feast import FeatureStore
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
FEATURE_SERVICE_NAME = "pm25_regression"
REGISTERED_MODEL_NAME = f"{FEATURE_SERVICE_NAME} model"
FEATURE_REPO = REPO_ROOT / "src/feature_repo"
PREDICTIONS_PATH = FEATURE_REPO / "data/predictions.parquet"
WEEKLY_PREDICTIONS_PATH = FEATURE_REPO / "data/batch_predictions.parquet"
ALLl_PREDICTIONS_PATH = FEATURE_REPO / "data/all_predictions.parquet"
Regressor = XGBRegressor | CatBoostRegressor | LGBMRegressor


def create_feature_store() -> FeatureStore:
    store = FeatureStore(
        repo_path=str(FEATURE_REPO),
        fs_yaml_file=FEATURE_REPO / "feature_store.yaml",
    )
    feast.mlflow.init(store=store)
    feast.mlflow.autolog()
    return store
