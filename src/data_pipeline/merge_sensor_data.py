import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
DATA_DIR = REPO_ROOT / "src/feature_repo/data"


def merge_sensor_data() -> pd.DataFrame:
    historical_sensor_df_path = DATA_DIR / "cleaned_historical_sensor_data.parquet"
    daily_sensor_df_path = DATA_DIR / "daily_sensor_data.parquet"

    historical_sensor_df = pd.read_parquet(historical_sensor_df_path)
    daily_sensor_df = pd.read_parquet(daily_sensor_df_path)
    daily_sensor_df["day"] = daily_sensor_df["date"].dt.date

    daily_sensor_df = daily_sensor_df[
        daily_sensor_df["day"] == datetime.today().date()
    ]

    median_today = (
        daily_sensor_df.groupby("day")["pm25"]
        .median()
        .to_frame()
        .reset_index()
        .rename(columns={"day": "date"})
    )
    median_today = median_today.assign(
        created_at=datetime.now(timezone.utc),
        country=os.environ.get("COUNTRY"),
        city=os.environ.get("CITY"),
        street=os.environ.get("STREET"),
        lat=float(os.environ["LAT"]),
        lon=float(os.environ["LON"]),
    )
    median_today["date"] = pd.to_datetime(median_today["date"], utc=True)
    df = pd.concat([historical_sensor_df, median_today], ignore_index=True)
    df.to_parquet(historical_sensor_df_path)
    return df


def main() -> None:
    load_dotenv(ENV_PATH, verbose=True)
    merge_sensor_data()


if __name__ == "__main__":
    main()