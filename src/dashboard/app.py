import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from util import (
    build_aqi_forecast_chart,
    create_feature_store,
    line_chart,
    load_entity_df,
    load_historical_data,
    make_mock_metrics,
    mean_std_chart,
)

st.set_page_config(layout="wide")

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
DATA_DIR = REPO_ROOT / "src/feature_repo/data"
FORECAST_PATH = DATA_DIR / "batch_predictions.parquet"
HINDCAST_PATH = DATA_DIR / "predictions.parquet"
METRICS_PATH = DATA_DIR / "monitoring_metrics.parquet"
REPORTS_PATH = REPO_ROOT / "evidently_artifacts/html_reports"

PLOTS_PER_ROW = 3
ROLLING_WINDOW = 7
DRIFT_THRESHOLD = 0.5
DATA_QUALITY_TABLE_ROWS = 10

column_colors = [
    "#636EFA",  # blue
    "#EF553B",  # red
    "#00CC96",  # green
    "#AB63FA",  # purple
    "#FFA15A",  # orange
    "#19D3F3",  # cyan
]

# AQI categories: (lower, upper, label, color)
AQI_BANDS = [
    (0, 50, "Good", "#00e400"),
    (50, 100, "Moderate", "#ffff00"),
    (100, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (150, 200, "Unhealthy", "#ff0000"),
    (200, 300, "Very Unhealthy", "#8f3f97"),
    (300, 500, "Hazardous", "#7e0023"),
]

band_labels = [band[2] for band in AQI_BANDS]
band_colors = {band[2]: band[3] for band in AQI_BANDS}
band_edges = [AQI_BANDS[0][0]] + [band[1] for band in AQI_BANDS]

weather_cols = [
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
]
target_cols = ["predicted_pm25", "pm25"]
all_cols = weather_cols + target_cols

regression_text_metrics = [
    "mean_error_mean",
    "mape_mean",
    "rmse",
    "mae_mean",
    "r2_score",
    "abs_max_error",
]

column_type_metrics = {
    "Numerical": "column_count_num",
    "Categorical": "column_count_cat",
    "Datetime": "column_count_datetime",
    "Text": "column_count_text",
}

summary_dataset_metrics = [
    "row_count",
    "column_count",
    "column_count_num",
    "column_count_cat",
    "column_count_datetime",
    "column_count_text",
    "missing_count",
    "missing_share",
]
summary_column_metrics = [f"{c}__missing_count" for c in all_cols] + [
    f"{c}__missing_share" for c in all_cols
]
summary_table_metrics = summary_dataset_metrics + summary_column_metrics
summary_graph_metrics = ["row_count", "missing_share"] + [
    f"{c}__missing_share" for c in all_cols
]

statistics_table_metrics = [
    f"{c}__{s}" for c in all_cols for s in ("min", "max", "mean", "std")
]
statistics_graph_metrics = [f"{c}__{s}" for c in all_cols for s in ("mean", "std")]

quantile_metrics = [f"{c}__{q}" for c in all_cols for q in ("q25", "q50", "q75")]
data_quality_metrics = [
    "duplicated_row_count",
    "duplicated_columns_count",
    "almost_duplicated_columns_count",
    "almost_constant_columns_count",
    "empty_rows_count",
    "empty_columns_count",
    "constant_columns_count",
]

drift_column_metrics = [f"{c}__drift" for c in all_cols]

MOCK_SEED = 42
MOCK_N = 30
MOCK_MISSING_SHARE = 0.2

# (typical mean, typical std) per column
col_profiles = {
    "temperature_2m_mean": (27.0, 0.8),
    "precipitation_sum": (9.0, 7.0),
    "wind_speed_10m_max": (10.5, 2.7),
    "wind_direction_10m_dominant": (202.0, 35.0),
    "predicted_pm25": (24.0, 4.0),
    "pm25": (16.0, 5.0),
}

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

historical_df = load_historical_data(store=store, entity_df=entity_df)
forecast_df = pd.read_parquet(FORECAST_PATH)
hindcast_df = pd.read_parquet(HINDCAST_PATH)

html_reports_paths = sorted(
    REPORTS_PATH.glob("*.html"), key=lambda p: p.name, reverse=True
)

# actual pm25, model prediction and the "tomorrow = today" baseline per day
history_df = (
    historical_df[["date", "pm25"]]
    .sort_values("date")
    .assign(naive_pm25=lambda d: d.pm25.shift(1))
)
eval_df = (
    hindcast_df[["date", "predicted_pm25"]]
    .merge(history_df, on="date", how="inner")
    .dropna()
    .sort_values("date")
)
eval_df["error"] = eval_df.predicted_pm25 - eval_df.pm25
eval_df["model_ae"] = eval_df.error.abs()
eval_df["naive_ae"] = (eval_df.pm25 - eval_df.naive_pm25).abs()

metrics_df = pd.read_parquet(METRICS_PATH)

metrics_df = make_mock_metrics(
    col_profiles=col_profiles,
    data_quality_metrics=data_quality_metrics,
    n=MOCK_N,
    seed=MOCK_SEED,
    mock_missing_share=MOCK_MISSING_SHARE,
)

home, data_summary, data_stats, data_drift, regression, reports = st.tabs(
    ["home", "data_summary", "data_stats", "data_drift", "regression", "reports"]
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

    with st.container(border=True, width="stretch"):
        day_bands = pd.cut(
            historical_df.pm25,
            bins=band_edges,
            labels=band_labels,
            include_lowest=True,
        )
        band_counts = day_bands.value_counts()
        band_counts = band_counts[band_counts > 0].reindex(band_labels).dropna()
        band_fig = go.Figure(
            go.Pie(
                labels=list(band_counts.index),
                values=band_counts.values,
                marker=dict(colors=[band_colors[label] for label in band_counts.index]),
                sort=False,
            )
        )
        band_fig.update_layout(title="DAYS BY AQI BAND")
        st.plotly_chart(band_fig, width="stretch")

with data_summary:
    latest = metrics_df.sort_values("report_date").iloc[-1]

    rows_latest_col, rows_total_col, missing_col = st.columns(3)

    with rows_latest_col:
        with st.container(border=True, width="stretch"):
            st.metric(label="Rows (latest)", value=f"{latest.row_count:.0f}")
            st.caption(latest.report_date.date())

    with rows_total_col:
        with st.container(border=True, width="stretch"):
            st.metric(
                label="Rows (cumulative)", value=f"{metrics_df.row_count.sum():.0f}"
            )
            st.caption(f"{len(metrics_df)} reports")

    with missing_col:
        with st.container(border=True, width="stretch"):
            st.metric(
                label="Missing share (latest)", value=f"{latest.missing_share:.1%}"
            )
            st.caption(latest.report_date.date())

    types_col, pie_col = st.columns(2)

    with types_col:
        st.caption("Column types")
        for col, (label, metric_name) in zip(
            st.columns(len(column_type_metrics)), column_type_metrics.items()
        ):
            with col:
                with st.container(border=True, width="stretch"):
                    st.metric(label=label, value=f"{latest[metric_name]:.0f}")

    with pie_col:
        with st.container(border=True, width="stretch"):
            missing_by_column = pd.Series(
                {c: latest[f"{c}__missing_count"] for c in all_cols}
            )
            if missing_by_column.sum() == 0:
                st.success("No missing values in the latest report")
            else:
                st.plotly_chart(
                    px.pie(
                        names=missing_by_column.index,
                        values=missing_by_column.values,
                        title="Missing values by column",
                    )
                )

    with st.container(border=True, width="stretch"):
        st.plotly_chart(
            line_chart(
                df=metrics_df, column="pm25__missing_share", color=column_colors[1]
            )
        )

    st.subheader("Data quality")
    st.dataframe(
        metrics_df.sort_values("report_date", ascending=False)[
            ["report_date"] + data_quality_metrics
        ].head(DATA_QUALITY_TABLE_ROWS),
        hide_index=True,
        width="stretch",
    )

with data_stats:
    for i, (col_name, color) in enumerate(zip(all_cols, column_colors)):
        if i % PLOTS_PER_ROW == 0:
            row_cols = st.columns(PLOTS_PER_ROW)

        with row_cols[i % PLOTS_PER_ROW]:  # type: ignore
            with st.container(border=True, width="stretch"):
                st.plotly_chart(
                    mean_std_chart(df=metrics_df, metric_name=col_name, color=color)
                )

with data_drift:
    for i, (metric_name, color) in enumerate(zip(drift_column_metrics, column_colors)):
        if i % PLOTS_PER_ROW == 0:
            row_cols = st.columns(PLOTS_PER_ROW)

        with row_cols[i % PLOTS_PER_ROW]:  # type: ignore
            with st.container(border=True, width="stretch"):
                st.plotly_chart(
                    line_chart(df=metrics_df, column=metric_name, color=color)
                )

    with st.container(border=True, width="stretch"):
        heatmap_fig = go.Figure(
            go.Heatmap(
                z=metrics_df[drift_column_metrics].T.values,
                x=metrics_df.report_date,
                y=all_cols,
                zmin=0,
                zmax=1,
                colorscale=[
                    [0, "#00CC96"],
                    [DRIFT_THRESHOLD, "#00CC96"],
                    [DRIFT_THRESHOLD, "#EF553B"],
                    [1, "#EF553B"],
                ],
                texttemplate="%{z:.2f}",
                colorbar=dict(title="DRIFT"),
            )
        )
        heatmap_fig.update_layout(
            title=f"DRIFT SCORE (THRESHOLD {DRIFT_THRESHOLD})",
            xaxis_title="DATE",
        )
        st.plotly_chart(heatmap_fig, width="stretch")

    bar_col, drift_pie_col = st.columns([2, 1])

    with bar_col:
        with st.container(border=True, width="stretch"):
            drifted_fig = px.bar(
                metrics_df,
                x="report_date",
                y="drifted_columns_count",
                color_discrete_sequence=[column_colors[1]],
                title="DRIFTED COLUMNS",
            )
            drifted_fig.update_layout(xaxis_title="DATE", yaxis_title="COUNT")
            st.plotly_chart(drifted_fig, width="stretch")

    with drift_pie_col:
        with st.container(border=True, width="stretch"):
            latest_report = metrics_df.sort_values("report_date").iloc[-1]
            drifted = int(latest_report.drifted_columns_count)
            drift_fig = go.Figure(
                go.Pie(
                    labels=["Drifted", "Stable"],
                    values=[drifted, len(all_cols) - drifted],
                    marker=dict(colors=["#EF553B", "#00CC96"]),
                    sort=False,
                )
            )
            drift_fig.update_layout(
                title=f"LATEST REPORT ({latest_report.report_date.date()})"
            )
            st.plotly_chart(drift_fig, width="stretch")

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

    for i, (metric_name, color) in enumerate(
        zip(regression_text_metrics, column_colors)
    ):
        if i % PLOTS_PER_ROW == 0:
            row_cols = st.columns(PLOTS_PER_ROW)

        with row_cols[i % PLOTS_PER_ROW]:  # type: ignore
            with st.container(border=True, width="stretch"):
                st.plotly_chart(
                    line_chart(df=metrics_df, column=metric_name, color=color)
                )

    scatter_col, hist_col = st.columns(2)

    with scatter_col:
        with st.container(border=True, width="stretch"):
            axis_range = [
                eval_df[["pm25", "predicted_pm25"]].min().min(),
                eval_df[["pm25", "predicted_pm25"]].max().max(),
            ]
            scatter_fig = go.Figure()
            scatter_fig.add_trace(
                go.Scatter(
                    x=eval_df.pm25,
                    y=eval_df.predicted_pm25,
                    mode="markers",
                    marker=dict(color=column_colors[0]),
                    name="DAYS",
                )
            )
            scatter_fig.add_trace(
                go.Scatter(
                    x=axis_range,
                    y=axis_range,
                    mode="lines",
                    line=dict(color="gray", dash="dash"),
                    name="Y = X",
                )
            )
            scatter_fig.update_layout(
                title="PREDICTED VS ACTUAL",
                xaxis_title="ACTUAL PM2.5",
                yaxis_title="PREDICTED PM2.5",
            )
            st.plotly_chart(scatter_fig)

    with hist_col:
        with st.container(border=True, width="stretch"):
            hist_fig = px.histogram(
                eval_df,
                x="error",
                nbins=20,
                color_discrete_sequence=[column_colors[1]],
                title="ERROR DISTRIBUTION (PREDICTED - ACTUAL)",
            )
            hist_fig.update_layout(xaxis_title="ERROR", yaxis_title="DAYS")
            st.plotly_chart(hist_fig)

    with st.container(border=True, width="stretch"):
        rolling_mae = (
            eval_df.set_index("date")[["model_ae", "naive_ae"]]
            .rolling(ROLLING_WINDOW, min_periods=1)
            .mean()
        )
        mae_fig = go.Figure()
        mae_fig.add_trace(
            go.Scatter(
                x=rolling_mae.index,
                y=rolling_mae.model_ae,
                mode="lines+markers",
                line=dict(color=column_colors[0]),
                name="MODEL",
            )
        )
        mae_fig.add_trace(
            go.Scatter(
                x=rolling_mae.index,
                y=rolling_mae.naive_ae,
                mode="lines+markers",
                line=dict(color=column_colors[3], dash="dash"),
                name="TOMORROW = TODAY",
            )
        )
        mae_fig.update_layout(
            title=f"{ROLLING_WINDOW}-DAY ROLLING MAE",
            xaxis_title="DATE",
            yaxis_title="MAE",
        )
        st.plotly_chart(mae_fig, width="stretch")

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
