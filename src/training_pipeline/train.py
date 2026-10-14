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
        mlflow_client = store.mlflow.mlflow
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

        train_dataset_df = pd.concat([Xtrain, ytrain], axis=1)
        val_dataset_df = pd.concat([Xval, yval], axis=1)
        train_dataset = mlflow_client.data.from_pandas(
            train_dataset_df,
            targets=target[0],
            name="train",
        )
        val_dataset = mlflow_client.data.from_pandas(
            val_dataset_df,
            targets=target[0],
            name="val",
        )
        mlflow_client.log_inputs(
            [train_dataset, val_dataset],
            contexts=["training", "validation"],
            tags_list=[None, None],
        )

        model.fit(Xtrain, ytrain)
        print(f"Fit {model.__class__.__name__}!")

        model_params = (
            model.get_all_params()
            if isinstance(model, CatBoostRegressor)
            else model.get_params()
        )
        mlflow_client.log_params(model_params)

        train_rmse = root_mean_squared_error(ytrain, model.predict(Xtrain))
        val_rmse = root_mean_squared_error(yval, model.predict(Xval))
        train_r2 = r2_score(ytrain, model.predict(Xtrain))
        val_r2 = r2_score(yval, model.predict(Xval))

        print(f"\tRMSE: \n\t\tTrain -> {train_rmse: .2f}\n\t\tVal -> {val_rmse: .2f}")
        print(f"\tR2: \n\t\tTrain -> {train_r2: .2f}\n\t\tVal -> {val_r2: .2f}")
        print()

        mlflow_client.log_metrics(
            {
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "train_r2": train_r2,
                "val_r2": val_r2,
            }
        )

        input_example = Xtrain.iloc[[0]]
        output_example = model.predict(input_example)
        signature = mlflow_client.models.infer_signature(
            input_example,
            output_example,
        )
        if isinstance(model, CatBoostRegressor):
            mlflow_client.catboost.log_model(
                model,
                name=MODEL_NAME,
                input_example=input_example,
                signature=signature,
            )
            # Feast's model wrapper does not map the CatBoost flavor yet.
            feast.mlflow.log_params(model.get_all_params())#_log_required_features()
        else:
            flavor = (
                "xgboost"
                if isinstance(model, XGBRegressor)
                else "lightgbm"
                if isinstance(model, LGBMRegressor)
                else "sklearn"
            )
            if flavor == "sklearn":
                store.mlflow.log_model(
                    model,
                    artifact_path=FEATURE_SERVICE_NAME,
                    flavor=flavor,
                    input_example=input_example,
                    signature=signature,
                    skops_trusted_types=SKLEARN_SKOPS_TRUSTED_TYPES,
                )
            elif flavor == "lightgbm":
                store.mlflow.log_model(
                    model,
                    artifact_path=FEATURE_SERVICE_NAME,
                    flavor=flavor,
                    input_example=input_example,
                    signature=signature,
                    skops_trusted_types=LIGHTGBM_SKOPS_TRUSTED_TYPES,
                )
            else:
                store.mlflow.log_model(
                    model,
                    artifact_path=FEATURE_SERVICE_NAME,
                    flavor=flavor,
                    input_example=input_example,
                    signature=signature,
                )

        mlflow_client.log_dict(
            {target[0]: float(output_example[0])},
            "examples/output_example.json",
        )


def main() -> None:
    warnings.filterwarnings("ignore")
    pd.set_option("display.max_columns", None)
    store = create_feature_store()
    entity_df = load_entity_df()

    models = [
        XGBRegressor(random_state=23),
        LGBMRegressor(random_state=25),
        RandomForestRegressor(random_state=23),
        HistGradientBoostingRegressor(random_state=23),
        CatBoostRegressor(random_seed=23, verbose=False),
    ]
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with store.mlflow.start_run(f"baseline_{run_timestamp}"):
        for model in models:
            train_and_evaluate_model(store, entity_df, model)


if __name__ == "__main__":
    main()
