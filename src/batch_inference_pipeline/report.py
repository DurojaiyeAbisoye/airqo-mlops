import warnings
from datetime import datetime, timezone
from pathlib import Path

import feast.mlflow
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.core.datasets import Regression
from evidently.metrics.data_quality import DatasetCorrelations
from evidently.metrics.regression import MeanError
from evidently.presets import DataDriftPreset, DataSummaryPreset, RegressionPreset
from evidently.ui.workspace import Project, RemoteWorkspace
from feast import FeatureStore
from util import create_feature_store

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
FEATURE_REPO = REPO_ROOT / "src/feature_repo"
FEATURE_SERVICE_NAME = "pm25_regression"
CHAMPION_MODEL_NAME = f"{FEATURE_SERVICE_NAME}_champion"

ENTITY_COLUMNS = ["city", "country", "street", "date"]
FEATURES = [
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "predicted_pm25",
]
TARGET = "pm25"
COLUMNS = ENTITY_COLUMNS + FEATURES + [TARGET] 


# fetch reference dataset
def fetch_reference_dataset(store: FeatureStore) -> pd.DataFrame:

    mlflow_client = feast.mlflow.MlflowClient(store.mlflow._tracking_uri)
    feast.mlflow.set_tracking_uri(store.mlflow._tracking_uri)

    model_version = mlflow_client.get_model_version_by_alias(
        name=CHAMPION_MODEL_NAME,
        alias="champion",
    )
    run_id = model_version.run_id
    local_path = mlflow_client.download_artifacts(
        run_id, "datasets/reference_dataset.parquet"
    )
    return pd.read_parquet(local_path)


def get_workspace_and_project(
    workspace_host="http://127.0.0.1:8000", project_name=FEATURE_SERVICE_NAME
) -> tuple[RemoteWorkspace, Project]:

    workspace = RemoteWorkspace(base_url=workspace_host)
    try:
        project = workspace.search_project(project_name=project_name)[0]
    except:
        project = workspace.create_project(
            name=project_name, description=f"{FEATURE_SERVICE_NAME} monitoring project"
        )
    return workspace, project


def monitor_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> None:

    workspace, project = get_workspace_and_project()
    report = Report(
        metrics=[
            DataSummaryPreset(),
            DataDriftPreset(),
            RegressionPreset(),
            DatasetCorrelations(),
            MeanError(error_plot=False, error_dist=False, error_normality=True),
        ],
        include_tests=True,
    )

    numerical_features = [
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "wind_direction_10m_dominant",
        "predicted_pm25",
        "pm25",
    ]

    data_definition = DataDefinition(
        id_column=None,
        numerical_columns=numerical_features,
        regression=[Regression(target="pm25", prediction="predicted_pm25")],
    )

    reference_data = Dataset.from_pandas(
        data=reference_df, data_definition=data_definition
    )
    current_data = Dataset.from_pandas(data=current_df, data_definition=data_definition)

    run_eval = report.run(
        current_data=current_data,
        reference_data=reference_data,
        timestamp=datetime.now(tz=timezone.utc),
        name=f"{FEATURE_SERVICE_NAME} report",
    )

    run_eval.save_html(
        filename=f"{REPO_ROOT}/evidently_artifacts/html_reports/report-{datetime.now().strftime('%Y-%m-%d')}.html"
    )
    run_eval.save_json(
        filename=f"{REPO_ROOT}/evidently_artifacts/json_reports/report-{datetime.now().strftime('%Y-%m-%d')}.json"
    )
    workspace.add_run(project_id=project.id, run=run_eval)


def main() -> None:

    warnings.filterwarnings("ignore")
    store = create_feature_store()
    
    reference_df = fetch_reference_dataset(store=store)[COLUMNS]
    current_df = pd.read_parquet(
        f"{FEATURE_REPO}/data/predictions.parquet", columns=COLUMNS
    )
    
    latest_actual_date = current_df.dropna(subset=["pm25"])["date"].max()
    cutoff = latest_actual_date - pd.Timedelta(days=14)
    
    current_df = current_df[(current_df.date > cutoff) & (current_df.date <= latest_actual_date)]
    current_df = current_df.dropna(subset=["pm25"])
    
    monitor_drift(reference_df=reference_df, current_df=current_df)


if __name__ == "__main__":
    main()
