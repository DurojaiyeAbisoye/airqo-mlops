import os

import openmeteo_requests
import pandas as pd


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
