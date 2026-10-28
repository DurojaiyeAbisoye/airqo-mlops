from pathlib import Path

import feast.mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from feast import FeatureStore

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
FEATURE_SERVICE_NAME = "pm25_regression"
REGISTERED_MODEL_NAME = f"{FEATURE_SERVICE_NAME} model"
FEATURE_REPO = REPO_ROOT / "src/feature_repo"
PREDICTIONS_PATH = FEATURE_REPO / "data/predictions.parquet"
WEEKLY_PREDICTIONS_PATH = FEATURE_REPO / "data/batch_predictions.parquet"
ALLl_PREDICTIONS_PATH = FEATURE_REPO / "data/all_predictions.parquet"


ENTITY_COLUMNS = ["city", "country", "street", "date"]
FEATURES = [
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
]
TARGET = "pm25"

Y_MIN, Y_MAX = 1, 500
LINE_WIDTH = 4
MARKER_SIZE = 10


def create_feature_store() -> FeatureStore:
    store = FeatureStore(
        repo_path=str(FEATURE_REPO),
        fs_yaml_file=FEATURE_REPO / "feature_store.yaml",
    )
    feast.mlflow.init(store=store)
    feast.mlflow.autolog()
    return store


def load_entity_df() -> pd.DataFrame:
    entity_df = pd.read_parquet(
        FEATURE_REPO / "data/cleaned_historical_sensor_data.parquet"
    )[ENTITY_COLUMNS]
    return entity_df


def load_historical_data(store: FeatureStore, entity_df: pd.DataFrame) -> pd.DataFrame:
    return (
        store.get_historical_features(
            entity_df=entity_df,
            features=store.get_feature_service(FEATURE_SERVICE_NAME),
            filter_by_created_timestamp=True,
        )
        .to_df()
        .sort_values("date")
        .reset_index(drop=True)
    )


def build_aqi_forecast_chart(AQI_BANDS, historical_df, forecast_df, hindcast_df):
    line_chart_fig = go.Figure()

    for low, high, label, color in AQI_BANDS:
        line_chart_fig.add_hrect(
            y0=max(low, Y_MIN),
            y1=min(high, Y_MAX),
            fillcolor=color,
            opacity=0.25,
            line_width=0,
            layer="below",
            annotation_text=label,
            annotation_position="right",
            annotation_font_size=11,
        )

    # Historical: many points, so a bold line with no markers
    line_chart_fig.add_trace(
        go.Scatter(
            x=historical_df.date,
            y=historical_df.pm25,
            mode="lines",
            name="Historical (true values)",
            line=dict(color="#111827", width=LINE_WIDTH - 1),
        )
    )

    # Forecast: short, so show markers too
    line_chart_fig.add_trace(
        go.Scatter(
            x=forecast_df.date,
            y=forecast_df.predicted_pm25,
            mode="lines+markers",
            name="Next week's forecast",
            line=dict(color="#1d4ed8", width=LINE_WIDTH, dash="dash"),
            marker=dict(
                size=MARKER_SIZE,
                color="#1d4ed8",
                line=dict(color="white", width=1.5),
            ),
        )
    )

    # Hindcast: magenta, bold, with markers
    line_chart_fig.add_trace(
        go.Scatter(
            x=hindcast_df.date,
            y=hindcast_df.predicted_pm25,
            mode="lines+markers",
            name="Hindcast (predicted)",
            line=dict(color="#c026d3", width=LINE_WIDTH),
            marker=dict(
                size=MARKER_SIZE,
                color="#c026d3",
                line=dict(color="white", width=1.5),
            ),
        )
    )

    line_chart_fig.update_layout(
        title="PM 2.5: Historical, Hindcast and Forecast",
        xaxis_title="Date",
        yaxis_title="PM 2.5 (AQI)",
        legend_title="Legend",
        hovermode="x unified",
        margin=dict(r=240),  # room for the band labels on the right
    )

    line_chart_fig.update_yaxes(
        type="log",
        range=[np.log10(Y_MIN), np.log10(Y_MAX)],
        tickvals=[1, 5, 10, 20, 50, 100, 200, 500],
        ticktext=["1", "5", "10", "20", "50", "100", "200", "500"],
    )

    return line_chart_fig


def line_chart(df: pd.DataFrame, column: str, color: str) -> go.Figure:
    label = column.replace("_", " ").upper()

    fig = go.Figure(
        go.Scatter(
            x=df["report_date"],
            y=df[column],
            mode="lines+markers",
            line=dict(color=color),
            name=label,
        )
    )
    fig.update_layout(
        title=label,
        xaxis_title="DATE",
        yaxis_title=label,
    )
    return fig
