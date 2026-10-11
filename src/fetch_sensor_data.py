import os
import argparse
import requests
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime, timezone
from pprint import pprint
import pandas as pd


def get_pm25(sensor_url: str, 
            sensor_api_key: str,
            country: str,
            city: str,
            street: str,
            lat: float,
            lon: float):

    url = f"{sensor_url}/?token={sensor_api_key}"
    response = requests.get(url)
    response_data = response.json()['data']
    created_at = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    return pd.DataFrame(
        [
            {
                "date": created_at,
                "pm25": float(response_data["iaqi"]["pm25"]["v"]),
                "created_at": created_at,
                "country": country,
                "city": city,
                "street": street,
                "lat": lat,
                "lon": lon,
            }
        ]
    )

def main() -> pd.DataFrame:
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path, verbose=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor-url", "--sensor_url", default=os.environ["AQICN_URL"])
    parser.add_argument("--sensor-token", "--sensor_token", default=os.environ["AQICN_TOKEN"])
    parser.add_argument("--country", default=os.environ["COUNTRY"])
    parser.add_argument("--city", default=os.environ["CITY"])
    parser.add_argument("--street", default=os.environ["STREET"])
    parser.add_argument("--lat", type=float, default=os.environ["LAT"])
    parser.add_argument("--lon", type=float, default=os.environ["LON"])
    args = parser.parse_args()

    return get_pm25(
        sensor_url=args.sensor_url,
        sensor_api_key=args.sensor_token,
        country=args.country,
        city=args.city,
        street=args.street,
        lat=args.lat,
        lon=args.lon,
    )

    
if __name__ == "__main__":
    main()
