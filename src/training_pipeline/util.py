import pandas as pd
from datetime import timedelta
from sklearn.model_selection import TimeSeriesSplit


def train_test_split(
    df: pd.DataFrame, event_column: str, num_test_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:

    df = df.copy().sort_values(event_column, ascending=True)

    end = df.event_timestamp.max()
    test_start = end - timedelta(days=num_test_days)

    train_val_df = df[df.event_timestamp <= test_start]
    test_df = df[df.event_timestamp > test_start]

    return train_val_df, test_df


def train_val_split(
    df: pd.DataFrame, num_test_days: int, n_splits: int
) -> pd.DataFrame:
    df = df.copy()
    tscv = TimeSeriesSplit(test_size=num_test_days, n_splits=n_splits)
    df["fold"] = -1

    for fold, (_, val_idx) in enumerate(tscv.split(df)):
        df.iloc[val_idx, fold] = fold

    return df

