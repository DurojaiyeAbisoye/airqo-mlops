from types import ModuleType

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, ScalarFormatter


def plot_air_quality_forecast(
    country: str | None,
    city: str | None,
    street: str | None,
    df: pd.DataFrame,
    file_path: str,
    hindcast: bool = False,
) -> ModuleType:
    _, ax = plt.subplots(figsize=(10, 6))
    day = pd.to_datetime(df["date"]).dt.date
    ax.plot(
        day,
        df["predicted_pm25"],
        label="Predicted PM2.5",
        color="red",
        linewidth=2,
        marker="o",
        markersize=5,
        markerfacecolor="blue",
    )

    ax.set_yscale("log")
    ax.set_yticks([0, 10, 25, 50, 100, 250, 500])
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    ax.set_ylim(bottom=1)

    ax.set_xlabel("Date")
    ax.set_title(f"PM2.5 Predicted (Logarithmic Scale) for {country} {city}, {street}")
    ax.set_ylabel("PM2.5")

    colors = ["green", "yellow", "orange", "red", "purple", "darkred"]
    labels = [
        "Good",
        "Moderate",
        "Unhealthy for Some",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    ]
    ranges = [(0, 49), (50, 99), (100, 149), (150, 199), (200, 299), (300, 500)]
    for color, (start, end) in zip(colors, ranges):
        ax.axhspan(start, end, color=color, alpha=0.3)

    patches = [
        Patch(color=colors[i], label=f"{labels[i]}: {ranges[i][0]}-{ranges[i][1]}")
        for i in range(len(colors))
    ]
    legend1 = ax.legend(
        handles=patches,
        loc="upper right",
        title="Air Quality Categories",
        fontsize="x-small",
    )

    if len(df.index) > 11:
        every_x_tick = len(df.index) / 10
        ax.xaxis.set_major_locator(MultipleLocator(every_x_tick))

    plt.xticks(rotation=45)

    if hindcast:
        ax.plot(
            day,
            df["pm25"],
            label="Actual PM2.5",
            color="black",
            linewidth=2,
            marker="^",
            markersize=5,
            markerfacecolor="grey",
        )
        ax.legend(loc="upper left", fontsize="x-small")
        ax.add_artist(legend1)

    plt.tight_layout()
    plt.savefig(file_path)
    return plt
