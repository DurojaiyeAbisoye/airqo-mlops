import os
from pathlib import Path

import feast.mlflow
import pandas as pd
from dotenv import load_dotenv
from feast import FeatureStore

from util import plot_air_quality_forecast

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
FEATURE_REPO = REPO_ROOT / "src/feature_repo"
FEATURE_VIEW_NAME = "air_quality_fv"
PREDICTIONS_PATH = FEATURE_REPO / "data/batch_predictions.parquet"
HINDCAST_PLOT_PATH = REPO_ROOT / "images/plots/hindcast.png"


def create_feature_store() -> FeatureStore:
    store = FeatureStore(
        repo_path=str(FEATURE_REPO),
        fs_yaml_file=FEATURE_REPO / "feature_store.yaml",
    )
    feast.mlflow.init(store=store)
    return store


def create_hindcast_df(store: FeatureStore) -> pd.DataFrame:
    feature_view = store.get_feature_view(FEATURE_VIEW_NAME)
    entity_columns = ["city", "country", "street", "date"]
    entity_df = pd.read_parquet(
        FEATURE_REPO / "data/cleaned_historical_sensor_data.parquet"
    )[entity_columns]
    entity_df = entity_df.rename(columns={"date": "event_timestamp"})

    air_quality_df = (
        store.get_historical_features(
            entity_df=entity_df,
            features=[f"{feature_view.name}:pm25"],
        )
        .to_df()
        .sort_values(by="event_timestamp")
        .reset_index(drop=True)
    )
    monitoring_df = pd.read_parquet(
        PREDICTIONS_PATH,
        columns=["date", "predicted_pm25", "days_before_forecast_day"],
    )
    predictions_df = monitoring_df[monitoring_df["days_before_forecast_day"] == 1]
    outcome_df = air_quality_df[["event_timestamp", "pm25"]].rename(
        columns={"event_timestamp": "date"}
    )
    return pd.merge(predictions_df, outcome_df, on="date").sort_values("date")


def main() -> None:
    load_dotenv(ENV_PATH, verbose=True)
    country = os.environ.get("COUNTRY")
    city = os.environ.get("CITY")
    street = os.environ.get("STREET")

    store = create_feature_store()
    hindcast_df = create_hindcast_df(store)
    if hindcast_df.empty:
        return

    HINDCAST_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_air_quality_forecast(
        country=country,
        city=city,
        street=street,
        df=hindcast_df,
        file_path=str(HINDCAST_PLOT_PATH),
        hindcast=True,
    )


if __name__ == "__main__":
    main()
