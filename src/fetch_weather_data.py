import os
import argparse
from pathlib import Path
import openmeteo_requests
import pandas as pd
from dotenv import load_dotenv
from .fetch_city_data import fetch_lon_lat


def fetch_open_meteo_df(
    url: str,
    start_date: str,
    end_date: str,
    daily_features: list[str],
) -> pd.DataFrame:

    country = os.environ.get("COUNTRY")
    city = os.environ.get("CITY")
    street = os.environ.get("STREET")
    location = os.environ.get("LOCATION")

    lon, lat = fetch_lon_lat(location)
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
        for i, name in enumerate(daily_features):
            daily = response.Daily()
            response_df[name] = daily.Variables(i).ValuesAsNumpy() 
        response_df["created_at"] = response_df.date.copy()
        response_df = response_df.assign(
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

    env_path = Path(__file__).parent.parent / ".env"
    _ = load_dotenv(env_path, verbose=True)

    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--url", type=str, help="url to fetch data from")
    _ = parser.add_argument(
        "--start_date", type=str, help="start date to use when fetching data"
    )
    _ = parser.add_argument(
        "--end_date", type=str, help="end date to use when fetching data"
    )
    _ = parser.add_argument(
        "--daily_features", type=str, nargs="+", help="list of daily features to fetch"
    )
    _ = parser.add_argument("--filename", type=str, help="filename")
    _ = parser.add_argument("--save_path", type=str, help="save_path")

    args = parser.parse_args()
    weather_df = fetch_open_meteo_df(
        url=args.url,
        start_date=args.start_date,
        end_date=args.end_date,
        daily_features=args.daily_features,
    )

    weather_df.to_parquet(f"{args.save_path}/{args.filename}.parquet")


if __name__ == "__main__":
    main()
