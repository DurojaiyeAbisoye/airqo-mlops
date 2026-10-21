from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
DATA_DIR = REPO_ROOT / "src/feature_repo/data"


def merge_sensor_data() -> pd.DataFrame:
    historical_sensor_df_path = DATA_DIR / "cleaned_historical_sensor_data.parquet"
    daily_sensor_df_path = DATA_DIR / "daily_sensor_data.parquet"

    historical_sensor_df = pd.read_parquet(historical_sensor_df_path)
    daily_sensor_df = pd.read_parquet(daily_sensor_df_path)

    df = pd.concat(
        [historical_sensor_df, daily_sensor_df], ignore_index=True
    ).drop_duplicates(subset=["date"], keep="last")

    df.to_parquet(historical_sensor_df_path)
    return df


def main() -> None:
    load_dotenv(ENV_PATH, verbose=True)
    merge_sensor_data()


if __name__ == "__main__":
    main()
