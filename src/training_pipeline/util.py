import pandas as pd
from datetime import timedelta
from sklearn.model_selection import TimeSeriesSplit
from feast import FeatureStore
from pathlib import Path
import feast.mlflow

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_REPO = REPO_ROOT / "src/feature_repo"
FEATURE_SERVICE_NAME = "pm25_regression"
MODEL_NAME = FEATURE_SERVICE_NAME
ENTITY_COLUMNS = ["city", "country", "street", "date"]


def create_feature_store() -> FeatureStore:
    store = FeatureStore(
        repo_path=str(FEATURE_REPO),
        fs_yaml_file=FEATURE_REPO / "feature_store.yaml",
    )
    feast.mlflow.init(store)
    return store


def load_entity_df() -> pd.DataFrame:
    entity_df = pd.read_parquet(
        FEATURE_REPO / "data/cleaned_historical_sensor_data.parquet"
    )[ENTITY_COLUMNS]
    return entity_df.rename(columns={"date": "event_timestamp"})

def load_historical_data(store: FeatureStore, entity_df: pd.DataFrame):
    df = (
        store.get_historical_features(
            entity_df=entity_df,
            features=store.get_feature_service(FEATURE_SERVICE_NAME),
            filter_by_created_timestamp=True,
        )
        .to_df()
        .sort_values("event_timestamp")
        .reset_index(drop=True)
    )
    return df

def train_test_split(
    df: pd.DataFrame, event_column: str, num_test_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:

    df = df.copy().sort_values(event_column, ascending=True)

    end = df.event_timestamp.max()
    test_start = end - timedelta(days=num_test_days)

    train_val_df = df[df.event_timestamp <= test_start]
    test_df = df[df.event_timestamp > test_start]

    return train_val_df, test_df

