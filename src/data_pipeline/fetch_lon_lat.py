import os
from pathlib import Path
from geopy.geocoders import Nominatim
from dotenv import load_dotenv


def fetch_lon_lat(location: str):
    geolocator = Nominatim(user_agent="ada")
    location_coordinate = geolocator.geocode(location)
    return location_coordinate.longitude, location_coordinate.latitude


if __name__ == "__main__":

    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path, verbose=True)

    location = os.getenv("LOCATION")
    print(location)

    print(fetch_lon_lat(location))
