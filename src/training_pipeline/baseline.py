import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feast.mlflow
import pandas as pd
from catboost import CatBoostRegressor
from feast import FeatureStore
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from util import create_feature_store, load_entity_df
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_REPO = REPO_ROOT / "src/feature_repo"
FEATURE_SERVICE_NAME = "pm25_regression"
MODEL_NAME = FEATURE_SERVICE_NAME
ENTITY_COLUMNS = ["city", "country", "street", "date"]
SKLEARN_SKOPS_TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]
LIGHTGBM_SKOPS_TRUSTED_TYPES = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMRegressor",
]


def split_historical_df(
    historical_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_end_date = historical_df["event_timestamp"].max()
    test_start_date = test_end_date - timedelta(days=30)

    train_df = historical_df[historical_df.event_timestamp <= test_start_date]
    test_df = historical_df[
        historical_df.event_timestamp.between(
            test_start_date, test_end_date, inclusive="right"
        )
    ]
    return train_df, test_df


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
        train_df, test_df = split_historical_df(historical_df)

        target = ["pm25"]
        features = train_df.drop(
            columns=["city", "country", "street", "event_timestamp"] + target
        ).columns.tolist()

        Xtrain = train_df[features]
        Xtest = test_df[features]
        ytrain = train_df[target].ffill()
        ytest = test_df[target].ffill()

        model.fit(Xtrain, ytrain)
        print(f"Fit {model.__class__.__name__}!")

        train_dataset_df = pd.concat([Xtrain, ytrain], axis=1)
        test_dataset_df = pd.concat([Xtest, ytest], axis=1)
        test_dataset_df["predicted_pm25"] = model.predict(Xtest)

        train_dataset_df_path = f"{FEATURE_REPO}/data/train_dataset.parquet"
        test_dataset_df_path = f"{FEATURE_REPO}/data/test_dataset.parquet"

        train_dataset_df.to_parquet(train_dataset_df_path, index=False)
        test_dataset_df.to_parquet(test_dataset_df_path, index=False)

        train_dataset = feast.mlflow.data.from_pandas(
            train_dataset_df,
            targets=target[0],
            name="train",
            source=train_dataset_df_path,
        )
        test_dataset = feast.mlflow.data.from_pandas(
            test_dataset_df,
            targets=target[0],
            name="test",
            source=test_dataset_df_path,
        )
        feast.mlflow.log_inputs(
            [train_dataset, test_dataset],
            contexts=["training", "testing"],
            tags_list=[None, None],
        )

        model_params = (
            model.get_all_params()
            if isinstance(model, CatBoostRegressor)
            else model.get_params()
        )
        feast.mlflow.log_params(model_params)

        train_rmse = root_mean_squared_error(ytrain, model.predict(Xtrain))
        test_rmse = root_mean_squared_error(ytest, model.predict(Xtest))
        train_r2 = r2_score(ytrain, model.predict(Xtrain))
        test_r2 = r2_score(ytest, model.predict(Xtest))

        print(f"\tRMSE: \n\t\tTrain -> {train_rmse: .2f}\n\t\tTest -> {test_rmse: .2f}")
        print(f"\tR2: \n\t\tTrain -> {train_r2: .2f}\n\t\tTest -> {test_r2: .2f}")
        print()

        feast.mlflow.log_metrics(
            {
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_r2": train_r2,
                "test_r2": test_r2,
            }
        )

        input_example = Xtrain.iloc[[0]]
        output_example = model.predict(input_example)
        signature = feast.mlflow.models.infer_signature(
            input_example,
            output_example,
        )
        if isinstance(model, CatBoostRegressor):
            store.mlflow.mlflow.catboost.log_model(
                model,
                name=MODEL_NAME,
                signature=signature,
            )
        elif isinstance(model, XGBRegressor):
            feast.mlflow.xgboost.log_model(
                model,
                artifact_path=FEATURE_SERVICE_NAME,
                signature=signature,
            )

        elif isinstance(model, LGBMRegressor):
            feast.mlflow.lightgbm.log_model(
                model,
                artifact_path=FEATURE_SERVICE_NAME,
                signature=signature,
                skops_trusted_types=LIGHTGBM_SKOPS_TRUSTED_TYPES,
            )
        else:
            feast.mlflow.sklearn.log_model(
                model,
                artifact_path=FEATURE_SERVICE_NAME,
                signature=signature,
                skops_trusted_types=SKLEARN_SKOPS_TRUSTED_TYPES,
            )


def main() -> None:
    warnings.filterwarnings("ignore")
    pd.set_option("display.max_columns", None)
    store = create_feature_store()
    entity_df = load_entity_df()

    models = [
        RandomForestRegressor(random_state=23),
        HistGradientBoostingRegressor(random_state=23),
        XGBRegressor(random_state=23),
        LGBMRegressor(random_state=25),
        CatBoostRegressor(random_seed=23, verbose=False),
    ]
    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M%:SZ")
    with store.mlflow.start_run(f"baseline_{run_timestamp}"):
        for model in models:
            train_and_evaluate_model(store, entity_df, model)


if __name__ == "__main__":
    main()
