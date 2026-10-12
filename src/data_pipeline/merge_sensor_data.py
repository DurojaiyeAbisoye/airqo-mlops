import os
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv


env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path, verbose=True)

historical_sensor_df_path = f'{Path(__file__).parent.parent}/feature_repo/data/cleaned_historical_sensor_data.parquet'
daily_sensor_df_path = f'{Path(__file__).parent.parent}/feature_repo/data/daily_sensor_data.parquet'

historical_sensor_df = pd.read_parquet(historical_sensor_df_path)
daily_sensor_df = pd.read_parquet(daily_sensor_df_path)
daily_sensor_df['day'] = daily_sensor_df.date.dt.date

daily_sensor_df = daily_sensor_df[daily_sensor_df.day == datetime.today().date()]

median_today = daily_sensor_df.groupby('day')['pm25'].median().to_frame().reset_index().rename(columns={'day' : 'date'})
median_today = median_today.assign(
    created_at = datetime.now(timezone.utc),
    country = os.environ.get("COUNTRY"),
    city = os.environ.get("CITY"),
    street = os.environ.get("STREET"),
    lat = float(os.environ["LAT"]),
    lon = float(os.environ["LON"])
        
)
median_today['date'] = pd.to_datetime(median_today.date, utc = True)
columns = historical_sensor_df.columns.tolist()
df = pd.concat([historical_sensor_df, median_today], ignore_index=True)
df.to_parquet(f'{Path(__file__).parent.parent}/feature_repo/data/cleaned_historical_sensor_data.parquet')