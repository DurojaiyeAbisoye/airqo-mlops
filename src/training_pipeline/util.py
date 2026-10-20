from datetime import timedelta
from pathlib import Path

import feast.mlflow
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from feast import FeatureStore
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_REPO = REPO_ROOT / "src/feature_repo"
FEATURE_SERVICE_NAME = "pm25_regression"
MODEL_NAME = FEATURE_SERVICE_NAME
ENTITY_COLUMNS = ["city", "country", "street", "date"]
FEATURES = [
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
]
TARGET = ["pm25"]


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


def load_historical_data(
    store: FeatureStore, entity_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = (
        store.get_historical_features(
            entity_df=entity_df,
            features=store.get_feature_service(FEATURE_SERVICE_NAME),
            filter_by_created_timestamp=True,
        )
        .to_df()
        .sort_values("event_timestamp")
        .reset_index(drop=True)
    )

    test_end_date = df.event_timestamp.max()
    test_start_date = test_end_date - timedelta(days=30)

    trainval_df = df[df.event_timestamp <= test_start_date]
    test_df = df[
        df.event_timestamp.between(test_start_date, test_end_date, inclusive="right")
    ]
    return df, trainval_df, test_df


def make_cv(test_size: int, gap: int, n_splits: int):
    return TimeSeriesSplit(test_size=test_size, n_splits=n_splits, gap=gap)


def build_training_config(
    feature_service_name: str,
    features: list[str],
    target: list[str],
    df: pd.DataFrame,
    trainval_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tscv: TimeSeriesSplit,
    test_size: int,
    gap: int,
):
    training_config = {
        "feature_service": feature_service_name,
        "features": features,
        "target": target,
        "metric": "rmse",
        "data": {
            "trainval_start": df["event_timestamp"].min().isoformat(),
            "trainval_end": trainval_df["event_timestamp"].max().isoformat(),
            "test_start": test_df["event_timestamp"].min().isoformat(),
            "test_end": test_df["event_timestamp"].max().isoformat(),
        },
        "cv": {
            "splitter": type(tscv).__name__,
            "n_splits": tscv.get_n_splits(),
            "test_size": test_size,
            "gap": gap,
            "max_train_size": None,
        },
    }
    return training_config


def tune_cv(
    trial: optuna.trial.Trial,
    store: FeatureStore,
    cv: TimeSeriesSplit,
    model: XGBRegressor | CatBoostRegressor | LGBMRegressor,
    X: pd.DataFrame,
    y: pd.Series,
) -> float:
    with store.mlflow.start_run(f"optuna trial {trial.number}", nested=True):
        model_params = model.get_params()
        feast.mlflow.log_params(model_params)

        train_rmse_scores = []
        val_rmse_scores = []
        train_r2_scores = []
        val_r2_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            with store.mlflow.start_run(f"Fold: {fold + 1}", nested=True):
                Xtrain = X.iloc[train_idx]
                Xval = X.iloc[val_idx]
                ytrain = y.iloc[train_idx]
                yval = y.iloc[val_idx]

                model.fit(Xtrain, ytrain)

                ytrain_pred = np.asarray(model.predict(Xtrain))
                yval_pred = np.asarray(model.predict(Xval))

                rmse_train = root_mean_squared_error(ytrain, ytrain_pred)
                rmse_val = root_mean_squared_error(yval, yval_pred)

                r2_train = r2_score(ytrain, ytrain_pred)
                r2_val = r2_score(yval, yval_pred)

                train_rmse_scores.append(rmse_train)
                val_rmse_scores.append(rmse_val)

                train_r2_scores.append(r2_train)
                val_r2_scores.append(r2_val)

                print(
                    f"Fold {fold + 1}/{cv.get_n_splits()} | "
                    f"Train RMSE: {rmse_train:.4f} | "
                    f"Validation RMSE: {rmse_val:.4f} |"
                    f"Train R2: {r2_train:.4f} |"
                    f"Val R2: {r2_val:.4f} |"
                )

                feast.mlflow.log_metrics(
                    {
                        "Train RMSE": rmse_train,
                        "Train R2": r2_train,
                        "Val RMSE": rmse_val,
                        "Val R2": r2_val,
                    }
                )
                trial.report(rmse_val, step=fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        train_rmse_mean = np.mean(train_rmse_scores)
        val_rmse_mean = np.mean(val_rmse_scores)

        train_r2_mean = np.mean(train_r2_scores)
        val_r2_mean = np.mean(val_r2_scores)

        print(
            f"CV mean | Train RMSE: {train_rmse_mean:.4f} | "
            f"Validation RMSE: {val_rmse_mean:.4f} |"
            f"Train R2: {train_r2_mean:.4f} |"
            f"Val R2: {val_r2_mean:.4f}"
        )
        feast.mlflow.log_metrics(
            {
                "Mean Train RMSE": train_rmse_mean,
                "Mean Train R2": train_r2_mean,
                "Mean Val RMSE": val_rmse_mean,
                "Mean Val R2": val_r2_mean,
            }
        )
        return val_rmse_mean


def tune_objective(
    model_family: str,
    model_objective_fn,
    store: FeatureStore,
    trainval_df: pd.DataFrame,
    cv: TimeSeriesSplit,
    features: list[str],
    target: list[str],
) -> optuna.study.Study:
    with store.mlflow.start_run(f"hparam-tuning-{model_family}", nested=True):
        sampler = optuna.samplers.TPESampler(seed=101)
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(
            direction="minimize", sampler=sampler, pruner=pruner
        )

        study.optimize(
            lambda trial: model_objective_fn(
                trial, store, trainval_df, cv, features, target
            ),
            n_trials=10,
            n_jobs=1,
        )

        feast.mlflow.log_params(
            {f"best_{name}": value for name, value in study.best_params.items()}
        )
        feast.mlflow.log_metric("best_cv_val_rmse", study.best_value)
        feast.mlflow.log_metric("best_trial", study.best_trial.number)
    return study
