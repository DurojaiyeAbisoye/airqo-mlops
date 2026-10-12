import os
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv


env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path, verbose=True)

historical_weather_df_path = f'{Path(__file__).parent.parent}/feature_repo/data/weather_historical.parquet'
daily_weather_df_path = f'{Path(__file__).parent.parent}/feature_repo/data/weather_forecast.parquet'

historical_weather_df = pd.read_parquet(historical_weather_df_path)
daily_weather_df = pd.read_parquet(daily_weather_df_path)

print(historical_weather_df.tail())
print(daily_weather_df.head())


df = pd.concat([historical_weather_df, daily_weather_df], ignore_index=True).drop_duplicates(subset = ['date'])
print(df.tail())
df.to_parquet(f'{Path(__file__).parent.parent}/feature_repo/data/weather_historical.parquet')