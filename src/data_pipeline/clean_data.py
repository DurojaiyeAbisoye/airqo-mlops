import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
OUTPUT_PATH = (
    REPO_ROOT / "src/feature_repo/data/cleaned_historical_sensor_data.parquet"
)
DEFAULT_INPUT_PATH = (
    REPO_ROOT / "src/feature_repo/data/raw_historical_sensor_data.csv"
)


def clean_df(df_path: str, save_df: bool = False) -> pd.DataFrame:
    country = os.environ.get("COUNTRY")
    city = os.environ.get("CITY")
    street = os.environ.get("STREET")
    lat = float(os.environ["LAT"])
    lon = float(os.environ["LON"])

    df = pd.read_csv(df_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"median": "pm25"})[["date", "pm25"]]
    df["created_at"] = df["date"]
    df = df.assign(
        country=country,
        city=city,
        street=street,
        lat=lat,
        lon=lon,
    )
    if save_df:
        df.to_parquet(OUTPUT_PATH, index=False)
    return df


def main() -> None:
    load_dotenv(ENV_PATH, verbose=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--df_path",
        type=str,
        help="Raw sensor df path",
        default=str(DEFAULT_INPUT_PATH),
    )
    parser.add_argument("--save_df", action="store_true", help="Save df")
    args = parser.parse_args()
    clean_df(args.df_path, args.save_df)


if __name__ == "__main__":
    main()
