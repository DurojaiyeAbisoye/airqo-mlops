import os
from pathlib import Path

from dotenv import load_dotenv
from geopy.geocoders import Nominatim

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"


def fetch_lon_lat(location: str) -> tuple[float, float]:
    geolocator = Nominatim(user_agent="ada")
    location_coordinate = geolocator.geocode(location)
    if location_coordinate is None:
        raise ValueError(f"Could not find coordinates for {location!r}")
    return location_coordinate.longitude, location_coordinate.latitude


def main() -> None:
    load_dotenv(ENV_PATH, verbose=True)

    location = os.environ["LOCATION"]
    print(location)
    print(fetch_lon_lat(location))


if __name__ == "__main__":
    main()
