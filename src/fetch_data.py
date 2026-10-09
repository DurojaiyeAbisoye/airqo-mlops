import openmeteo_requests
import pandas as pd


def fetch_open_meteo_df(
    url: str,
    start_date: str,
    end_date: str,
    longitude: float,
    latitude: float,
    daily_features: list,
) -> pd.DataFrame:

    open_meteo = openmeteo_requests.Client()
    params = dict(
        start_date=start_date,
        end_date=end_date,
        longitude=longitude,
        latitude=latitude,
        daily=daily_features,
    )

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
            country="Nigeria",
            city="Port-Harcourt",
            state="Rivers State",
            street="Eremo Ogbagoro",
            lat=4.7896647,
            lon=6.9714602,
        )
        return response_df
    else:
        raise ValueError(f"response is none! Check your parameters.")
