# retrieve best model
# Get its feature service
# use that to get the features/batch inference data
# predict on that
# Save predictions to monitoring fv(add days b4 forecast column)
# plot and save forecast to repo
# create hindcast plot
# save plot to repo
import os
import feast.mlflow
from feast import FeatureStore
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

pd.set_option("display.max_columns", None)

feature_service_name = "pm25_regression"

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
MODEL_NAME = f"{feature_service_name} model"

load_dotenv(ENV_PATH, verbose=True)


store = FeatureStore(
    repo_path=f"{REPO_ROOT}/src/feature_repo/",
    fs_yaml_file=Path(f"{REPO_ROOT}/src/feature_repo/feature_store.yaml"),
)


feast.mlflow.init(store=store)

feast.mlflow.autolog()

with store.mlflow.start_run("inference"):

    mlflow_client = feast.mlflow.MlflowClient(store.mlflow._tracking_uri)
    feast.mlflow.set_tracking_uri(store.mlflow._tracking_uri)

    model_version = mlflow_client.get_model_version_by_alias(
        name=MODEL_NAME, alias="champion"
    )

    model_uri = mlflow_client.get_model_version_download_uri(
        name=MODEL_NAME, version=model_version.version
    )
    model = store.mlflow.load_model(model_uri)
    fs_name = store.mlflow.resolve_features(
        f"models:/{MODEL_NAME}/{model_version.aliases[0]}"
    )

    inference_entity_df = pd.DataFrame()
    start_date = datetime.now(tz=timezone.utc).date()
    end_date = start_date + timedelta(days=6)
    inference_entity_df["event_timestamp"] = pd.date_range(
        start_date,
        end_date,
        freq="1D",
        name="event_timestamp",
        normalize=True,
        tz=timezone.utc,
    )

    country = os.environ.get("COUNTRY")
    city = os.environ.get("CITY")
    street = os.environ.get("STREET")

    inference_entity_df = inference_entity_df.assign(
        country=country, city=city, street=street
    )

    batch_inference_df = (
        store.get_historical_features(
            entity_df=inference_entity_df,
            features=store.get_feature_service(fs_name),
            # filter_by_created_timestamp=True,
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
    batch_inference_df = batch_inference_df.assign(
        country=country,
        city=city,
        street=street,
        days_before_forecast_day=range(1, len(batch_inference_df) + 1),
    )

    save_path = f"{REPO_ROOT}/src/feature_repo/data/batch_predictions.parquet"
    if Path(save_path).exists():
        store.write_to_offline_store("monitoring_fv", df=batch_inference_df)
    else:
        batch_inference_df.to_parquet(save_path)
