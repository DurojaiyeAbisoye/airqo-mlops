import os
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent.parent / ".env"
print(env_path)
_ = load_dotenv(env_path, verbose=True)


def clean_df(df_path: str, save_df: bool = False) -> pd.DataFrame:

    country = os.environ.get("COUNTRY")
    city = os.environ.get("CITY")
    street = os.environ.get("STREET")
    lat = float(os.environ["LAT"])
    lon = float(os.environ["LON"])

    df = pd.read_csv(df_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"median": "pm25"})[["date", "pm25"]]
    df["created_at"] = df.date.copy()
    df = df.assign(
        country=country,
        city=city,
        street=street,
        lat=lat,
        lon=lon,
    )
    if save_df:
        df.to_parquet(
            f"{env_path.parent}/src/feature_repo/data/cleaned_historical_sensor_data.parquet", index=False
        )
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--df_path", type=str, help="Raw sensor df path", default=f'{env_path.parent}/src/feature_repo/data/raw_historical_sensor_data.csv')
    _ = parser.add_argument("--save_df", action="store_true", help="Save df")
    args = parser.parse_args()
    _ = clean_df(args.df_path, args.save_df) 