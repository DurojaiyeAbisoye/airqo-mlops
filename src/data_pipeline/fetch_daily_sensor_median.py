import argparse
import json
import math
import os
from pathlib import Path
from statistics import median

import pandas as pd
import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"


def fetch_yesterdays_sensor_data(
    country: str,
    city: str,
    street: str,
    lat: float,
    lon: float,
    station_id: int = 816424,
    save_path: str | None = None,
) -> None:
    if save_path is None:
        save_path = str(REPO_ROOT / "src/feature_repo/data/daily_sensor_data.parquet")

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

    yesterday = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(1, unit="D")
    url = (
        f"https://airnet.waqi.info/airnet/sse/historic/hourly/"
        f"{station_id}?specie=pm25"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    hourly_medians = []
    text = response.text.replace("\r\n", "\n").replace("\r", "\n")

    for event_block in text.split("\n\n"):
        data_lines = [
            line.partition(":")[2].lstrip()
            for line in event_block.splitlines()
            if line.startswith("data:")
        ]
        if not data_lines:
            continue

        payload = json.loads("\n".join(data_lines))

        if not (
            isinstance(payload, list)
            and payload
            and isinstance(payload[0], (int, float))
        ):
            continue

        timestamp = int(payload[0])
        index = 1

        while index < len(payload):
            values = payload[index]
            if not isinstance(values, list) or len(values) < 3:
                raise ValueError(f"Unexpected hourly data format: {values!r}")

            hour = pd.to_datetime(timestamp, unit="s", utc=True)
            hourly_median = values[2]

            if hour.normalize() == yesterday and hourly_median is not None:
                hourly_median = float(hourly_median)
                if math.isfinite(hourly_median):
                    hourly_medians.append(hourly_median)

            index += 1
            if index < len(payload):
                timestamp += int(payload[index])
                index += 1

    daily_median = median(hourly_medians) if hourly_medians else float("nan")

    yesterday_df = pd.DataFrame(
        [
            {
                "date": yesterday,
                "pm25": daily_median,
                "created_at": pd.Timestamp.now(tz="UTC"),
                "country": country,
                "city": city,
                "street": street,
                "lat": lat,
                "lon": lon,
            }
        ],
        columns=columns,
    )

    yesterday_df.to_parquet(parquet_path, index=False)


def main() -> None:
    load_dotenv(ENV_PATH, verbose=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--station-id", type=int, default=816424)
    parser.add_argument("--country", default=os.environ["COUNTRY"])
    parser.add_argument("--city", default=os.environ["CITY"])
    parser.add_argument("--street", default=os.environ["STREET"])
    parser.add_argument("--lat", type=float, default=os.environ["LAT"])
    parser.add_argument("--lon", type=float, default=os.environ["LON"])
    parser.add_argument(
        "--save-path",
        "--save_path",
        default=str(REPO_ROOT / "src/feature_repo/data/daily_sensor_data.parquet"),
    )
    args = parser.parse_args()

    fetch_yesterdays_sensor_data(
        country=args.country,
        city=args.city,
        street=args.street,
        lat=args.lat,
        lon=args.lon,
        station_id=args.station_id,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
