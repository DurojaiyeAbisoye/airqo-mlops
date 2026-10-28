from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

df = pd.read_parquet(REPO_ROOT / "src/feature_repo/data/monitoring_metrics.parquet")
weather_cols = [
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
]
target_cols = ["predicted_pm25", "pm25"]
all_cols = weather_cols + target_cols

regression_metrics = [
    "mean_error_mean",
    "mean_error_std",
    "mape_mean",
    "mape_std",
    "rmse",
    "mae_mean",
    "mae_std",
    "r2_score",
    "abs_max_error",
]

drift_dataset_metrics = ["drifted_columns_count", "drifted_columns_share"]
drift_column_metrics = [f"{c}__drift" for c in all_cols]
drift_metrics = drift_dataset_metrics + drift_column_metrics

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

categorized = set(
    ["report_date"]
    + regression_metrics
    + drift_metrics
    + summary_table_metrics
    + statistics_table_metrics
    + quantile_metrics
    + data_quality_metrics
)
