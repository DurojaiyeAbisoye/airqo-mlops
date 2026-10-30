from datetime import datetime, timedelta, timezone
from pathlib import Path

import feast.mlflow
import pandas as pd
from catboost import CatBoostRegressor
from feast import FeatureStore
from lightgbm import LGBMRegressor
from mlflow.entities.model_registry import ModelVersion
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"

PROJECT_NAME = "pm25_regression"
FEATURE_SERVICE_VERSION = 1
FEATURE_SERVICE_NAME = f"{PROJECT_NAME}_v{FEATURE_SERVICE_VERSION}"

FEATURE_REPO = REPO_ROOT / "src/feature_repo"
PREDICTIONS_PATH = FEATURE_REPO / "data/predictions.parquet"
WEEKLY_PREDICTIONS_PATH = FEATURE_REPO / "data/batch_predictions.parquet"
ALL_PREDICTIONS_PATH = FEATURE_REPO / "data/all_predictions.parquet"
Regressor = XGBRegressor | CatBoostRegressor | LGBMRegressor


def create_feature_store() -> FeatureStore:
    store = FeatureStore(
        repo_path=str(FEATURE_REPO),
        fs_yaml_file=FEATURE_REPO / "feature_store.yaml",
    )
    feast.mlflow.init(store=store)
    feast.mlflow.autolog()
    return store


def create_inference_entity_df(
    country: str | None,
    city: str | None,
    street: str | None,
) -> pd.DataFrame:
    start_date = datetime.now(tz=timezone.utc).date()
    end_date = start_date + timedelta(days=6)
    entity_df = pd.DataFrame(
        {
            "date": pd.date_range(
                start_date,
                end_date,
                freq="1D",
                name="date",
                normalize=True,
                tz=timezone.utc,
            )
        }
    )
    return entity_df.assign(country=country, city=city, street=street)