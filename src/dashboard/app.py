import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from util import (
    build_aqi_forecast_chart,
    create_feature_store,
    line_chart,
    load_entity_df,
    load_historical_data,
)

st.set_page_config(layout="wide")

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"

load_dotenv(ENV_PATH)
country = os.environ.get("COUNTRY")
city = os.environ.get("CITY")
street = os.environ.get("STREET")
lat = float(os.environ["LAT"])
lon = float(os.environ["LON"])

location_df = pd.DataFrame(
    data={
        "country": [country],
        "city": [city],
        "street": [street],
        "longitude": [lon],
        "latitude": [lat],
    }
)
st.title("pm25 regression")
st.write(REPO_ROOT)

store = create_feature_store()
entity_df = load_entity_df()

# AQI categories: (lower, upper, label, color)
AQI_BANDS = [
    (0, 50, "Good", "#00e400"),
    (50, 100, "Moderate", "#ffff00"),
    (100, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (150, 200, "Unhealthy", "#ff0000"),
    (200, 300, "Very Unhealthy", "#8f3f97"),
    (300, 500, "Hazardous", "#7e0023"),
]


historical_df = load_historical_data(store=store, entity_df=entity_df)
forecast_df = pd.read_parquet(
    REPO_ROOT / "src/feature_repo/data/batch_predictions.parquet"
)
hindcast_df = pd.read_parquet(REPO_ROOT / "src/feature_repo/data/predictions.parquet")


home, data_summary, data_stats, data_drift, regression, reports = st.tabs(
    ["home", "data_summary", "data_stats", "data_drift", "regression", "reports"]
)

reports_path = REPO_ROOT / "evidently_artifacts/html_reports"
html_reports_paths = sorted(
    reports_path.glob("*.html"), key=lambda p: p.name, reverse=True
)

with home:
    st.header("Sensors used")

    map_column, text_column = st.columns([4, 1], gap="xsmall")

    with map_column:
        st.map(
            data=location_df,
            longitude="longitude",
            latitude="latitude",
            width="stretch",
            height=330,
        )

    with text_column:
        with st.container(border=True, width="stretch"):
            st.metric(
                label="Last observed pm2.5",
                value=f"{historical_df.pm25.iloc[-1]:.2f} AQI",
                delta=f"{historical_df.pm25.iloc[-1] - historical_df.pm25.iloc[-2]:.2f}",
                delta_color="inverse",
            )
            st.caption(historical_df.date.iloc[-1].date())

        with st.container(border=True, width="stretch"):
            st.metric(
                label="Predicted pm2.5 for tomorrow",
                value=f"{forecast_df.predicted_pm25.iloc[0]:.2f} AQI",
                delta=f"{forecast_df.predicted_pm25.iloc[0] - historical_df.pm25.iloc[-1]:.2f}",
                delta_color="inverse",
            )
            st.caption(forecast_df.date.iloc[0].date())

    st.plotly_chart(
        build_aqi_forecast_chart(AQI_BANDS, historical_df, forecast_df, hindcast_df),
        width="stretch",
    )


with reports:
    if not html_reports_paths:
        st.info("No reports yet.")
    else:
        selected = st.selectbox(
            label="Evidently reports",
            options=html_reports_paths,
            format_func=lambda path: path.name,
            index=None,
            placeholder="Select a report",
        )
        if selected:
            st.iframe(selected, height=900)


metrics_df = pd.read_parquet(
    REPO_ROOT / "src/feature_repo/data/monitoring_metrics.parquet"
)

regression_text_metrics = [
    "mean_error_mean",
    "mape_mean",
    "rmse",
    "mae_mean",
    "r2_score",
    "abs_max_error",
]

rng = np.random.default_rng(42)
n = 30

# (typical mean, typical std) per column
col_profiles = {
    "temperature_2m_mean": (27.0, 0.8),
    "precipitation_sum": (9.0, 7.0),
    "wind_speed_10m_max": (10.5, 2.7),
    "wind_direction_10m_dominant": (202.0, 35.0),
    "predicted_pm25": (24.0, 4.0),
    "pm25": (16.0, 5.0),
}

data = {
    "report_date": pd.date_range("2026-01-04", periods=n, freq="W"),
    "row_count": rng.integers(13, 16, n).astype(float),
    "column_count": np.full(n, 6.0),
    "column_count_num": np.full(n, 6.0),
    "column_count_cat": np.zeros(n),
    "column_count_datetime": np.zeros(n),
    "column_count_text": np.zeros(n),
    "missing_count": np.zeros(n),
    "missing_share": rng.uniform(0, 0.03, n),
}

for col, (mu, sigma) in col_profiles.items():
    mean = mu + rng.normal(0, sigma * 0.2, n)
    std = np.abs(sigma + rng.normal(0, sigma * 0.1, n))
    data[f"{col}__mean"] = mean
    data[f"{col}__std"] = std
    data[f"{col}__min"] = mean - 1.8 * std
    data[f"{col}__max"] = mean + 1.8 * std
    data[f"{col}__missing_count"] = np.zeros(n)
    data[f"{col}__missing_share"] = rng.uniform(0, 0.03, n)
    data[f"{col}__drift"] = rng.uniform(0, 1, n)

data["drifted_columns_count"] = rng.integers(0, 4, n).astype(float)
data["drifted_columns_share"] = data["drifted_columns_count"] / 6

data["mean_error_mean"] = rng.normal(-3, 2, n)
data["mean_error_std"] = rng.uniform(3, 6, n)
data["mape_mean"] = rng.uniform(15, 35, n)
data["mape_std"] = rng.uniform(10, 25, n)
data["rmse"] = rng.uniform(4, 9, n)
data["mae_mean"] = rng.uniform(3, 8, n)
data["mae_std"] = rng.uniform(2, 5, n)
data["r2_score"] = rng.uniform(-0.5, 0.7, n)
data["abs_max_error"] = rng.uniform(10, 25, n)

metrics_df = pd.DataFrame(data)


with regression:
    cols = st.columns(6)

    latest_two_dates = metrics_df.report_date.nlargest(2)
    latest_date = latest_two_dates.iloc[0]
    previous_date = latest_two_dates.iloc[1] if len(latest_two_dates) > 1 else None

    latest_row = metrics_df[metrics_df.report_date == latest_date]
    previous_row = (
        metrics_df[metrics_df.report_date == previous_date]
        if previous_date is not None
        else None
    )

    for col, metric_name in zip(cols, regression_text_metrics):
        latest_value = latest_row[metric_name].iloc[0]
        previous_value = (
            previous_row[metric_name].iloc[0] if previous_row is not None else None
        )
        delta = latest_value - previous_value if previous_value is not None else None

        with col:
            with st.container(border=True, width="stretch"):
                st.metric(
                    label=metric_name.replace("_", " ").upper(),
                    value=f"{latest_value:.2f}",
                    delta=f"{delta:.2f}" if delta is not None else None,
                    delta_color="normal" if metric_name == "r2_score" else "inverse",
                )
                st.caption(f"latest: {latest_date.date()}")

    regression_colors = [
            "#636EFA",  # blue
            "#EF553B",  # red
            "#00CC96",  # green
            "#AB63FA",  # purple
            "#FFA15A",  # orange
            "#19D3F3",  # cyan
        ]
    
    PLOTS_PER_ROW = 3

    for i, (metric_name, color) in enumerate(zip(regression_text_metrics, regression_colors)):
        if i % PLOTS_PER_ROW == 0:
            row_cols = st.columns(PLOTS_PER_ROW)

        with row_cols[i % PLOTS_PER_ROW]: # type: ignore
            with st.container(border=True, width="stretch"):
                st.plotly_chart(
                    line_chart(df=metrics_df, column=metric_name, color=color)
                )