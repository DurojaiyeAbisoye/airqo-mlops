import argparse
import os
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
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"


# python src/data_pipeline/fetch_weather_data.py --url "https://archive-api.open-meteo.com/v1/archive" --start_date '2026-01-01' --end_date '2026-10-11' --filename 'weather_historical'


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
    daily = response.Daily()
    if daily is None:
        raise RuntimeError("Open-Meteo response did not include daily data.")

    response_df = pd.DataFrame()
    response_df["date"] = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=int(daily.Interval())),
        inclusive="left",
    )
    for i, name in enumerate(daily_features):
        variable = daily.Variables(i)
        if variable is None:
            raise RuntimeError(f"Open-Meteo response did not include {name!r}.")
        response_df[name] = variable.ValuesAsNumpy()

    return response_df.assign(
        created_at=response_df["date"],
        country=country,
        city=city,
        street=street,
        lat=lat,
        lon=lon,
    )


def main() -> None:
    load_dotenv(ENV_PATH, verbose=True)

    today = datetime.now(timezone.utc).date()
    env = os.environ.get

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        default=env("OPEN_METEO_FORECAST_URL"),
        help="url to fetch data from",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=today.isoformat(),
        help="start date to use when fetching data (default: today, UTC)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=(today + timedelta(days=6)).isoformat(),
        help="end date to use when fetching data (default: today + 6 days)",
    )
    parser.add_argument(
        "--daily_features",
        type=str,
        nargs="+",
        default=DEFAULT_DAILY_FEATURES,
        help="list of daily features to fetch",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="weather_forecast",
        help="filename",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(REPO_ROOT / "src/feature_repo/data"),
        help="folder to save into",
    )

    args = parser.parse_args()
    weather_df = fetch_open_meteo_df(
        url=args.url,
        start_date=args.start_date,
        end_date=args.end_date,
        daily_features=args.daily_features,
    )

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    weather_df.to_parquet(save_path / f"{args.filename}.parquet")


if __name__ == "__main__":
    main()
