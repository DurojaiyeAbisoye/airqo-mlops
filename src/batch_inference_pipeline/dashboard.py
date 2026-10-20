from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import (
    counter_panel,
    line_plot_panel,
    pie_plot_panel,
    text_panel,
)
from evidently.ui.workspace import Project


def create_dashboard(project: Project, title: str):
    """
    One Evidently project per model version.
    model_info: string from MLflow (version, train date, train/infer time).
    """
    p = project.dashboard
    p.add_panel(panel=text_panel(title=title))

    # ----- data summary -----
    p.add_panel(
        panel=counter_panel(
            title="Total Rows (sum)",
            values=[PanelMetric(metric="RowCount")],
            aggregation="sum",
            size="half",
        )
    )
    p.add_panel(
        panel=counter_panel(
            title="RowCount (last)",
            values=[PanelMetric(metric="RowCount")],
            aggregation="last",
            size="half",
        )
    )
    p.add_panel(
        panel=counter_panel(
            title="Column Count (last)",
            values=[
                PanelMetric(
                    metric="ColumnCount",
                )
            ],
            aggregation="last",
            size="half",
        )
    )
    p.add_panel(
        panel=counter_panel(
            title="Empty Rows / Cols (last)",
            values=[
                PanelMetric(metric="EmptyRowsCount", legend="empty rows"),
                PanelMetric(metric="EmptyColumnsCount", legend="empty cols"),
            ],
            aggregation="last",
            size="half",
        )
    )

    p.add_panel(
        panel=pie_plot_panel(
            title="Missing Values (last)",
            values=[PanelMetric(metric="DatasetMissingValueCount")],
            aggregation="last",
            size="half",
        )
    )
    p.add_panel(
        panel=line_plot_panel(
            title="Missing Count Over Time",
            values=[
                PanelMetric(
                    metric="DatasetMissingValueCount",
                    metric_labels={"value_type": "count"},
                    legend="count",
                ),
            ],
            size="half",
        )
    )
    p.add_panel(
        panel=line_plot_panel(
            title="Missing Share Over Time",
            values=[
                PanelMetric(
                    metric="DatasetMissingValueCount",
                    metric_labels={"value_type": "share"},
                    legend="share",
                ),
            ],
            size="half",
        )
    )

    p.add_panel(
        panel=pie_plot_panel(
            title="Drifted Columns Share (last)",
            values=[
                PanelMetric(
                    metric="DriftedColumnsCount",
                    metric_labels={"value_type": "share"},
                    legend="share",
                ),
            ],
            aggregation="last",
            size="half",
        )
    )
    p.add_panel(
        panel=line_plot_panel(
            title="Drifted Columns Count Over Time",
            values=[
                PanelMetric(
                    metric="DriftedColumnsCount",
                    metric_labels={"value_type": "count"},
                    legend="count",
                ),
            ],
            size="half",
        )
    )
    p.add_panel(
        panel=line_plot_panel(
            title="Drifted Columns Share Over Time",
            values=[
                PanelMetric(
                    metric="DriftedColumnsCount",
                    metric_labels={"value_type": "share"},
                    legend="share",
                ),
            ],
            size="half",
        )
    )

    for metric, labels in [
        ("MAE", {"value_type": "mean"}),
        ("MeanError", {"value_type": "mean"}),
        ("MAPE", {"value_type": "mean"}),
        ("RMSE", None),
        ("R2Score", None),
        ("AbsMaxError", None),
    ]:
        p.add_panel(
            panel=counter_panel(
                title=f"{metric} (last)",
                values=[
                    (
                        PanelMetric(metric=metric, metric_labels=labels)
                        if labels
                        else PanelMetric(metric=metric)
                    )
                ],
                aggregation="last",
                size="half",
            )
        )

    for metric in ["MAE", "MeanError", "MAPE"]:
        values = [
            PanelMetric(
                metric=metric,
                metric_labels={"value_type": "mean"},
                legend=f"{metric} mean",
            ),
        ]
        # MAPE std has scale bug vs mean — skip std for MAPE
        if metric != "MAPE":
            values.append(
                PanelMetric(
                    metric=metric,
                    metric_labels={"value_type": "std"},
                    legend=f"{metric} std",
                )
            )
        p.add_panel(
            panel=line_plot_panel(
                title=f"{metric} over time",
                values=values,
                size="half",
            )
        )

    for metric in ["RMSE", "R2Score", "AbsMaxError"]:
        p.add_panel(
            panel=line_plot_panel(
                title=f"{metric} over time",
                values=[PanelMetric(metric=metric, legend=metric)],
                size="half",
            )
        )

    project.save()
