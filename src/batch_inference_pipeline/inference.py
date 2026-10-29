import os
from datetime import datetime, timezone
from pathlib import Path

import feast.mlflow
import pandas as pd
from dotenv import load_dotenv
from feast import FeatureStore
from util import create_feature_store, create_inference_entity_df

pd.set_option("display.max_columns", None)

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
FEATURE_SERVICE_NAME = "pm25_regression"
FEATURE_REPO = REPO_ROOT / "src/feature_repo"
CHAMPION_MODEL_NAME = f"{FEATURE_SERVICE_NAME}_champion"

PREDICTIONS_PATH = (
    FEATURE_REPO / "data/predictions.parquet"
)  # daily targets + ground truth, grows
WEEKLY_PREDICTIONS_PATH = (
    FEATURE_REPO / "data/batch_predictions.parquet"
)  # this week's forecast, overwritten
ALL_PREDICTIONS_PATH = (
    FEATURE_REPO / "data/all_predictions.parquet"
)  # every day's full forecast, grows


def predict_batch(
    store: FeatureStore,
    country: str | None,
    city: str | None,
    street: str | None,
) -> pd.DataFrame:
    mlflow_client = feast.mlflow.MlflowClient(store.mlflow._tracking_uri)
    feast.mlflow.set_tracking_uri(store.mlflow._tracking_uri)

    model_version = mlflow_client.get_model_version_by_alias(
        name=CHAMPION_MODEL_NAME,
        alias="champion",
    )
    model_uri = mlflow_client.get_model_version_download_uri(
        name=CHAMPION_MODEL_NAME,
        version=model_version.version,
    )
    model = store.mlflow.load_model(model_uri).get_raw_model()
    feature_service_name = store.mlflow.resolve_features(
        f"models:/{CHAMPION_MODEL_NAME}/champion"
    )

    inference_entity_df = create_inference_entity_df(country, city, street)
    batch_inference_df = (
        store.get_historical_features(
            entity_df=inference_entity_df,
            features=store.get_feature_service(feature_service_name),
        )
        .to_df()
        .sort_values(by="date")
        .reset_index(drop=True)
        .drop(columns=["city", "country", "street", "date", "pm25"])
    )

    batch_inference_df["predicted_pm25"] = model.predict(batch_inference_df)
    batch_inference_df["date"] = inference_entity_df["date"]
    return batch_inference_df.assign(
        country=country,
        city=city,
        street=street,
        days_before_forecast_day=range(1, len(batch_inference_df) + 1),
    )


def save_inference_data(batch_inference_df, today, yesterday):
    # 1) Weekly: just this run's forecast, overwritten every time (no concat)
    batch_inference_df.to_parquet(WEEKLY_PREDICTIONS_PATH, index=False)

    # 2) All: append the whole forecast every day, tagged with the day it was made
    all_df = batch_inference_df.assign(run_date=today)
    if ALL_PREDICTIONS_PATH.exists():
        all_df = (
            pd.concat([pd.read_parquet(ALL_PREDICTIONS_PATH), all_df])
            .drop_duplicates(subset=["run_date", "date"], keep="last")
            .reset_index(drop=True)
        )
    all_df.to_parquet(ALL_PREDICTIONS_PATH, index=False)

    # 3) Daily: one row per day (1-day-ahead prediction) + backfilled ground truth
    prediction_today = batch_inference_df[
        (batch_inference_df.date == today)
        & (batch_inference_df.days_before_forecast_day == 1)
    ]
    if "pm25" not in prediction_today.columns:
        prediction_today = prediction_today.assign(pm25=pd.NA)

    # Start from today's row; if history exists, append to it
    predictions_df = prediction_today
    if PREDICTIONS_PATH.exists():
        predictions_df = (
            pd.concat([pd.read_parquet(PREDICTIONS_PATH), prediction_today])
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )

    # Backfill yesterday's actual pm25 (sensor file holds a single row)
    sensor_df = pd.read_parquet(FEATURE_REPO / "data/daily_sensor_data.parquet")
    if sensor_df.empty or pd.isna(sensor_df["pm25"].iloc[0]):
        print("No sensor pm25 yet, skipping backfill")
    else:
        sensor_date = sensor_df["date"].iloc[0]
        sensor_pm25 = sensor_df["pm25"].iloc[0]
        predictions_df.loc[predictions_df["date"] == sensor_date, "pm25"] = sensor_pm25

    predictions_df.to_parquet(PREDICTIONS_PATH, index=False)


def main() -> None:
    load_dotenv(ENV_PATH, verbose=True)
    country = os.environ.get("COUNTRY")
    city = os.environ.get("CITY")
    street = os.environ.get("STREET")
    store = create_feature_store()

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with store.mlflow.start_run(f"inference_{run_timestamp}"):
        batch_inference_df = predict_batch(store, country, city, street)

    today = pd.Timestamp.now(tz="UTC").normalize()
    yesterday = today - pd.Timedelta(days=1)

    save_inference_data(batch_inference_df, today, yesterday)


if __name__ == "__main__":
    main()
