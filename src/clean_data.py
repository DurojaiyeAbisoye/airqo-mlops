import pandas as pd


def clean_df(df: pd.DataFrame, save_df: bool = False) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df.date)
    df = df.rename(columns={"median": "pm25"})[["date", "pm25"]]
    df = df.set_index("date").resample(rule="1D").asfreq().reset_index()
    df["created_at"] = df.date.copy()
    df = df.assign(
        country="Nigeria",
        city="Port-Harcourt",
        state="Rivers State",
        street="Eremo Ogbagoro",
        lat=4.7896647,
        lon=6.9714602,
    )
    if save_df:
        df.to_parquet("src/feature_repo/data/cleaned-air-quality-data.parquet", index=False)
    return df
