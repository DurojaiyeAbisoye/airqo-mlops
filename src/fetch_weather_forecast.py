import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

import openmeteo_requests
import pandas as pd
from dotenv import load_dotenv

DEFAULT_DAILY_FEATURES = [
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
]


def fetch_open_meteo_df(
    url: str,
    start_date: str,
    end_date: str,
    daily_features: list[str],
) -> pd.DataFrame:

    country = os.environ.get("COUNTRY")
    city = os.environ.get("CITY")
    street = os.environ.get("STREET")
    lat = float(os.environ["LAT"])
    lon = float(os.environ["LON"])
    open_meteo = openmeteo_requests.Client()
    params = {
        "start_date": start_date,
        "end_date": end_date,
        "longitude": lon,
        "latitude": lat,
        "daily": daily_features,
    }

    response = open_meteo.weather_api(url=url, params=params)[0]
    if response:
        response_df = pd.DataFrame()
        response_df["date"] = pd.date_range(
            start=start_date, end=end_date, freq="1D", normalize=True, tz="UTC"
        )
        daily = response.Daily()
        for i, name in enumerate(daily_features):
            response_df[name] = daily.Variables(i).ValuesAsNumpy()
        response_df = response_df.assign(
            created_at=datetime.now(timezone.utc),
            country=country,
            city=city,
            street=street,
            lat=lat,
            lon=lon,
        )
        return response_df
    else:
        raise ValueError("response is none! Check your parameters.")


def main():

    env_path = Path(__file__).resolve().parent.parent / ".env"
    _ = load_dotenv(env_path, verbose=True)

    today = datetime.now(timezone.utc).date()
    env = os.environ.get

    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--url",
        type=str,
        default=env("OPEN_METEO_FORECAST_URL"),
        help="url to fetch data from",
    )
    _ = parser.add_argument(
        "--start_date",
        type=str,
        default=today.isoformat(),
        help="start date to use when fetching data (default: today, UTC)",
    )
    _ = parser.add_argument(
        "--end_date",
        type=str,
        default=(today + timedelta(days=6)).isoformat(),
        help="end date to use when fetching data (default: today + 6 days)",
    )
    _ = parser.add_argument(
        "--daily_features",
        type=str,
        nargs="+",
        default=env("DAILY_FEATURES", "").split() or DEFAULT_DAILY_FEATURES,
        help="list of daily features to fetch",
    )
    _ = parser.add_argument(
        "--filename",
        type=str,
        default=env("FORECAST_FILENAME", "weather_forecast"),
        help="filename",
    )
    _ = parser.add_argument(
        "--save_path",
        type=str,
        default=env("FORECAST_SAVE_DIR", "src/feature_repo/data"),
        help="folder to save into",
    )

    args = parser.parse_args()
    weather_df = fetch_open_meteo_df(
        url=args.url,
        start_date=args.start_date,
        end_date=args.end_date,
        daily_features=args.daily_features,
    )

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    weather_df.to_parquet(f"{args.save_path}/{args.filename}.parquet")


if __name__ == "__main__":
    main()
