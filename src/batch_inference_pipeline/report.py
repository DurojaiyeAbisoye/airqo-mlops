import os
import feast.mlflow
from pathlib import Path
from evidently import Report, Dataset, DataDefinition
from evidently.ui.workspace import RemoteWorkspace, Project
from evidently.core.datasets import Regression
from evidently.presets import DataDriftPreset, DataSummaryPreset, RegressionPreset
from evidently.metrics.regression import MeanError
from evidently.metrics.data_quality import DatasetCorrelations
from feast import FeatureStore
import pandas as pd
from datetime import datetime, timezone
import warnings
from dashboard import create_dashboard

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
FEATURE_REPO = REPO_ROOT / "src/feature_repo"
FEATURE_SERVICE_NAME = "pm25_regression"
MODEL_NAME = f"{FEATURE_SERVICE_NAME} model"


def create_feature_store() -> FeatureStore:
    store = FeatureStore(
        repo_path=str(FEATURE_REPO),
        fs_yaml_file=FEATURE_REPO / "feature_store.yaml",
    )
    feast.mlflow.init(store=store)
    return store


# fetch reference dataset
def fetch_reference_dataset(store: FeatureStore) -> pd.DataFrame:

    mlflow_client = feast.mlflow.MlflowClient(store.mlflow._tracking_uri)
    feast.mlflow.set_tracking_uri(store.mlflow._tracking_uri)

    model_version = mlflow_client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias="champion",
    )
    run_id = model_version.run_id
    run = feast.mlflow.get_run(run_id=run_id)

    input_datasets = run.inputs.dataset_inputs
    for dataset_input in input_datasets:
        dataset = dataset_input.dataset
        if dataset.name == "val":
            dataset_source = feast.mlflow.data.get_source(dataset).load()
            reference_df = pd.read_parquet(dataset_source)
    return reference_df  # type: ignore


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
    create_dashboard(project, FEATURE_SERVICE_NAME)


def main() -> None:

    warnings.filterwarnings("ignore")
    store = create_feature_store()
    reference_df = fetch_reference_dataset(store=store)
    current_df = pd.read_parquet(
        f"{FEATURE_REPO}/data/batch_predictions.parquet",
        columns=[
            "temperature_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max",
            "wind_direction_10m_dominant",
            "predicted_pm25",
        ],
    )
    current_df["pm25"] = current_df["predicted_pm25"] * 1.0103
    monitor_drift(reference_df=reference_df, current_df=current_df)


if __name__ == "__main__":
    main()
