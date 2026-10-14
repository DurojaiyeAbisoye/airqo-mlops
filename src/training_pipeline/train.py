import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feast.mlflow
import pandas as pd
from feast import FeatureStore
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_REPO = REPO_ROOT / "src/feature_repo"
FEATURE_SERVICE_NAME = "pm25_regression"
ENTITY_COLUMNS = ["city", "country", "street", "date"]


def create_feature_store() -> FeatureStore:
    store = FeatureStore(
        repo_path=str(FEATURE_REPO),
        fs_yaml_file=FEATURE_REPO / "feature_store.yaml",
    )
    feast.mlflow.init(store)
    return store


def load_entity_df() -> pd.DataFrame:
    entity_df = pd.read_parquet(
        FEATURE_REPO / "data/cleaned_historical_sensor_data.parquet"
    )[ENTITY_COLUMNS]
    return entity_df.rename(columns={"date": "event_timestamp"})


def split_historical_df(
    historical_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    last_date = historical_df["event_timestamp"].max()
    val_end_date = last_date - timedelta(days=7)
    train_end_date = val_end_date - timedelta(days=7)

    train_df = historical_df[historical_df.event_timestamp <= train_end_date]
    val_df = historical_df[
        (historical_df.event_timestamp > train_end_date)
        & (historical_df.event_timestamp <= val_end_date)
    ]
    test_df = historical_df[historical_df.event_timestamp > val_end_date]
    print(train_df.shape, val_df.shape, test_df.shape)
    return train_df, val_df, test_df


def train_and_evaluate_model(
    store: FeatureStore,
    entity_df: pd.DataFrame,
    model,
) -> None:
    with store.mlflow.start_run(nested=True):
        historical_df = (
            store.get_historical_features(
                entity_df=entity_df,
                features=store.get_feature_service(FEATURE_SERVICE_NAME),
            )
            .to_df()
            .sort_values(by="event_timestamp")
            .reset_index(drop=True)
        )
        print(f"{historical_df.shape = }")
        train_df, val_df, test_df = split_historical_df(historical_df)

        target = ["pm25"]
        features = train_df.drop(
            columns=["city", "country", "street", "event_timestamp"] + target
        ).columns.tolist()
        print(f"{features = }")
        print(f"{target = }")

        Xtrain = train_df[features]
        Xval = val_df[features]
        ytrain = train_df[target].ffill()
        yval = val_df[target].ffill()

        model.fit(Xtrain, ytrain)
        print(f"Fit {model.__class__.__name__}!")

        train_rmse = root_mean_squared_error(ytrain, model.predict(Xtrain))
        val_rmse = root_mean_squared_error(yval, model.predict(Xval))
        train_r2 = r2_score(ytrain, model.predict(Xtrain))
        val_r2 = r2_score(yval, model.predict(Xval))

        print(f"\tRMSE: \n\t\tTrain -> {train_rmse: .2f}\n\t\tVal -> {val_rmse: .2f}")
        print(f"\tR2: \n\t\tTrain -> {train_r2: .2f}\n\t\tVal -> {val_r2: .2f}")
        print()

        feast.mlflow.log_metrics(
            {
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "train_r2": train_r2,
                "val_r2": val_r2,
            }
        )


def main() -> None:
    warnings.filterwarnings("ignore")
    pd.set_option("display.max_columns", None)
    store = create_feature_store()
    entity_df = load_entity_df()

    feast.mlflow.sklearn.autolog()
    feast.mlflow.xgboost.autolog()
    feast.mlflow.lightgbm.autolog()

    models = [
        XGBRegressor(random_state=23),
        LGBMRegressor(random_state=25),
        RandomForestRegressor(random_state=23),
        HistGradientBoostingRegressor(random_state=23),
    ]
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with store.mlflow.start_run(f"baseline_{run_timestamp}"):
        for model in models:
            train_and_evaluate_model(store, entity_df, model)


if __name__ == "__main__":
    main()
