from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "src/feature_repo/data"


def merge_weather_data() -> pd.DataFrame:
    historical_weather_df_path = DATA_DIR / "weather_historical.parquet"
    daily_weather_df_path = DATA_DIR / "weather_forecast.parquet"

    historical_weather_df = pd.read_parquet(historical_weather_df_path)
    daily_weather_df = pd.read_parquet(daily_weather_df_path)

    df = pd.concat(
        [historical_weather_df, daily_weather_df], ignore_index=True
    ).drop_duplicates(subset=["date"])
    df.to_parquet(historical_weather_df_path)
    return df


def main() -> None:
    merge_weather_data()


if __name__ == "__main__":
    main()
