import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feast.mlflow
import pandas as pd
from dotenv import load_dotenv
from feast import FeatureStore

from util import plot_air_quality_forecast

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
FEATURE_SERVICE_NAME = "pm25_regression"
MODEL_NAME = f"{FEATURE_SERVICE_NAME} model"
FEATURE_REPO = REPO_ROOT / "src/feature_repo"
PREDICTIONS_PATH = FEATURE_REPO / "data/batch_predictions.parquet"
FORECAST_PLOT_PATH = REPO_ROOT / "images/plots/forecast.png"


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
            "event_timestamp": pd.date_range(
                start_date,
                end_date,
                freq="1D",
                name="event_timestamp",
                normalize=True,
                tz=timezone.utc,
            )
        }
    )
    return entity_df.assign(country=country, city=city, street=street)


def predict_batch(
    store: FeatureStore,
    country: str | None,
    city: str | None,
    street: str | None,
) -> pd.DataFrame:
    mlflow_client = feast.mlflow.MlflowClient(store.mlflow._tracking_uri)
    feast.mlflow.set_tracking_uri(store.mlflow._tracking_uri)

    model_version = mlflow_client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias="champion",
    )
    model_uri = mlflow_client.get_model_version_download_uri(
        name=MODEL_NAME,
        version=model_version.version,
    )
    model = store.mlflow.load_model(model_uri)
    feature_service_name = store.mlflow.resolve_features(
        f"models:/{MODEL_NAME}/{model_version.aliases[0]}"
    )

    inference_entity_df = create_inference_entity_df(country, city, street)
    batch_inference_df = (
        store.get_historical_features(
            entity_df=inference_entity_df,
            features=store.get_feature_service(feature_service_name),
        )
        .to_df()
        .sort_values(by="event_timestamp")
        .reset_index(drop=True)
    )
    batch_inference_df = batch_inference_df.drop(
        columns=["city", "country", "street", "event_timestamp", "pm25"]
    )
    batch_inference_df["predicted_pm25"] = model.predict(batch_inference_df)
    batch_inference_df["date"] = inference_entity_df["event_timestamp"]
    return batch_inference_df.assign(
        country=country,
        city=city,
        street=street,
        days_before_forecast_day=range(1, len(batch_inference_df) + 1),
    )


def save_predictions(predictions_df: pd.DataFrame) -> None:
    if PREDICTIONS_PATH.exists():
        keys = [
            "country",
            "city",
            "street",
            "date",
            "days_before_forecast_day",
        ]
        predictions_df = (
            pd.concat(
                [pd.read_parquet(PREDICTIONS_PATH), predictions_df],
                ignore_index=True,
            )
            .drop_duplicates(subset=keys, keep="last")
            .reset_index(drop=True)
        )
        temporary_path = PREDICTIONS_PATH.with_suffix(".tmp")
        predictions_df.to_parquet(temporary_path)
        temporary_path.replace(PREDICTIONS_PATH)
    else:
        predictions_df.to_parquet(PREDICTIONS_PATH)


def main() -> None:
    load_dotenv(ENV_PATH, verbose=True)
    country = os.environ.get("COUNTRY")
    city = os.environ.get("CITY")
    street = os.environ.get("STREET")
    store = create_feature_store()

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with store.mlflow.start_run(f"inference_{run_timestamp}"):
        batch_inference_df = predict_batch(store, country, city, street)
        FORECAST_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        plot_air_quality_forecast(
            country=country,
            city=city,
            street=street,
            df=batch_inference_df,
            file_path=str(FORECAST_PLOT_PATH),
        )
        save_predictions(batch_inference_df)


if __name__ == "__main__":
    main()
