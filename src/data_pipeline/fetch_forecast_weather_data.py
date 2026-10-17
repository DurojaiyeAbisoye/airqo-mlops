import os
from pathlib import Path
from dotenv import load_dotenv
from fetch_weather_data import fetch_open_meteo_df
from datetime import datetime, timedelta, timezone
import argparse

DEFAULT_DAILY_FEATURES = [
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
]
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"


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
