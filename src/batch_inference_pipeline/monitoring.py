import os
import feast.mlflow
from feast import FeatureStore
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
from util import plot_air_quality_forecast
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"

load_dotenv(ENV_PATH, verbose=True)
store = FeatureStore(
    repo_path=f"{REPO_ROOT}/src/feature_repo/",
    fs_yaml_file=Path(f"{REPO_ROOT}/src/feature_repo/feature_store.yaml"),
)

country = os.environ.get("COUNTRY")
city = os.environ.get("CITY")
street = os.environ.get("STREET")


feast.mlflow.init(store=store)

air_quality_fv_name = "air_quality_fv"
air_quality_fv = store.get_feature_view(air_quality_fv_name)

entity_columns = ["city", "country", "street", "date"]
entity_df = pd.read_parquet(
    f"{REPO_ROOT}/src/feature_repo/data/cleaned_historical_sensor_data.parquet"
)[entity_columns]
entity_df = entity_df.rename(columns={"date": "event_timestamp"})

air_quality_df = (
    store.get_historical_features(
        entity_df=entity_df, features=[f"{air_quality_fv.name}:pm25"]
    )
    .to_df()
    .sort_values(by="event_timestamp")
    .reset_index(drop=True)
)

monitoring_df = pd.read_parquet(
    f"{REPO_ROOT}/src/feature_repo/data/batch_predictions.parquet",
    columns=["date", "predicted_pm25", "days_before_forecast_day"],
)
preds_df = monitoring_df[monitoring_df.days_before_forecast_day == 1]
outcome_df = air_quality_df[["event_timestamp", "pm25"]].rename(
    columns={"event_timestamp": "date"}
)


hindcast_df = pd.merge(preds_df, outcome_df, on="date").sort_values("date")

if not hindcast_df.empty:
    plot_air_quality_forecast(
        city=city,
        street=street,
        country=country,
        df=hindcast_df,
        file_path=f"{REPO_ROOT}/images/plots/hindcast.png",
        hindcast=True,
    )
