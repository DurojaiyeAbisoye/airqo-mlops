import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from fetch_sensor_data import get_pm25


def fetch_hourly_sensor_data(
    save_path: str | None = None,
) -> pd.DataFrame:

    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path, verbose=True)

    if save_path is None:
        save_path = "src/feature_repo/data/daily_sensor_data.parquet"

    parquet_path = Path(save_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "date",
        "pm25",
        "created_at",
        "country",
        "city",
        "street",
        "lat",
        "lon",
    ]

    try:
        df = pd.read_parquet(parquet_path)

    except FileNotFoundError:
        df = pd.DataFrame(columns=columns)

    hourly_df = get_pm25(
        sensor_url=os.environ["AQICN_URL"],
        sensor_api_key=os.environ["AQICN_TOKEN"],
        country=os.environ["COUNTRY"],
        city=os.environ["CITY"],
        street=os.environ["STREET"],
        lat=float(os.environ["LAT"]),
        lon=float(os.environ["LON"]),
    )

    df = (
        pd.concat([df[columns], hourly_df[columns]], ignore_index=True)
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    df.to_parquet(parquet_path, index=False)
    print(df)
    return df


if __name__ == "__main__":
    fetch_hourly_sensor_data()
