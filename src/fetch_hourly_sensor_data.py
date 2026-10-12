import os
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from fetch_sensor_data import get_pm25


def fetch_hourly_sensor_data(
    sensor_url: str,
    sensor_token: str,
    country: str,
    city: str,
    street: str,
    lat: float,
    lon: float,
    save_path: str | None = None,
) -> pd.DataFrame:

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
        sensor_url=sensor_url,
        sensor_api_key=sensor_token,
        country=country,
        city=city,
        street=street,
        lat=lat,
        lon=lon,
    )

    df = (
        pd.concat([df[columns], hourly_df[columns]], ignore_index=True)
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    print(df)
    df.to_parquet(parquet_path, index=False)
    return df


def main() -> pd.DataFrame:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path, verbose=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor-url", "--sensor_url", default=os.environ["AQICN_URL"])
    parser.add_argument(
        "--sensor-token", "--sensor_token", default=os.environ["AQICN_TOKEN"]
    )
    parser.add_argument("--country", default=os.environ["COUNTRY"])
    parser.add_argument("--city", default=os.environ["CITY"])
    parser.add_argument("--street", default=os.environ["STREET"])
    parser.add_argument("--lat", type=float, default=os.environ["LAT"])
    parser.add_argument("--lon", type=float, default=os.environ["LON"])
    parser.add_argument("--save-path", "--save_path", default=None)
    args = parser.parse_args()

    return fetch_hourly_sensor_data(
        sensor_url=args.sensor_url,
        sensor_token=args.sensor_token,
        country=args.country,
        city=args.city,
        street=args.street,
        lat=args.lat,
        lon=args.lon,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
