from datetime import datetime, timedelta, timezone
from pathlib import Path

import feast.mlflow
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from feast import FeatureStore
from lightgbm import LGBMRegressor
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.models.model import ModelInfo
from sklearn.base import clone
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

Regressor = XGBRegressor | CatBoostRegressor | LGBMRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_REPO = REPO_ROOT / "src/feature_repo"
FEATURE_SERVICE_NAME = "pm25_regression"

CHAMPION_MODEL_NAME = f"{FEATURE_SERVICE_NAME}_champion"
CHALLENGER_MODEL_NAME = f"{FEATURE_SERVICE_NAME}_challenger"

CHALLENGER_ALIAS = "challenger"
CHAMPION_ALIAS = "champion"
PREVIOUS_ALIAS = "previous"

TEST_WINDOW_DAYS = 14  # held-out test window; equals the biweekly retrain interval
MIN_REL_IMPROVEMENT = 0.0  # challenger must beat champion RMSE by MORE than this fraction

ENTITY_COLUMNS = ["city", "country", "street", "date"]
FEATURES = [
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
]
TARGET = "pm25"

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
    test_start_date = test_end_date - timedelta(days=TEST_WINDOW_DAYS)

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
    target: str,
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
    model: Regressor,
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
                    f"Validation RMSE: {rmse_val:.4f} | "
                    f"Train R2: {r2_train:.4f} | "
                    f"Val R2: {r2_val:.4f}"
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
            f"Validation RMSE: {val_rmse_mean:.4f} | "
            f"Train R2: {train_r2_mean:.4f} | "
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
    target: str,
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
            n_trials=1,
            n_jobs=1,
        )

        feast.mlflow.log_params(
            {f"best_{name}": value for name, value in study.best_params.items()}
        )
        feast.mlflow.log_metric("best_cv_val_rmse", study.best_value)
        feast.mlflow.log_metric("best_trial", study.best_trial.number)
    return study


def log_selected_model(
    store: FeatureStore,
    model: Regressor,
    model_name: str,
    model_family: str,
    features_df: pd.DataFrame,
) -> ModelInfo:
    input_example = features_df.iloc[[0]]
    output_example = model.predict(input_example)
    signature = feast.mlflow.models.infer_signature(
        input_example,
        output_example,
    )

    if model_family == "catboost":
        return feast.mlflow.catboost.log_model(
            model,
            name=model_name,
            signature=signature,
        )
    elif model_family == "xgboost":
        return feast.mlflow.xgboost.log_model(
            model,
            name=model_name,
            signature=signature,
        )
    else:
        return feast.mlflow.lightgbm.log_model(
            model,
            name=model_name,
            signature=signature,
            skops_trusted_types=LIGHTGBM_SKOPS_TRUSTED_TYPES,
        )


def register_model(
    store: FeatureStore,
    model_info: ModelInfo,
    model_name: str,
    alias: str,
    model_family: str,
    val_rmse: float,
) -> ModelVersion:
    model_version = store.mlflow.register_model(
        model_info.model_uri,
        name=model_name,
    )
    print("Model Registered!")

    mlflow_client = feast.mlflow.MlflowClient(store.mlflow._tracking_uri)
    for key, value in {
        "model_family": model_family,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": model_info.run_id,
        "val_rmse": str(val_rmse),
        "feast.feature_service": FEATURE_SERVICE_NAME,
    }.items():
        mlflow_client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key=key,
            value=value,
        )
    mlflow_client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=model_version.version,
    )
    print("Model Aliased!")
    # Return a fresh copy so the tags and alias set above are visible on the object.
    return mlflow_client.get_model_version(model_name, model_version.version)


def get_model_version(client, model_name: str, alias: str) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except MlflowException as e:
        if e.error_code == "RESOURCE_DOES_NOT_EXIST":
            return None
        raise


def load_model(store, client, model_version: ModelVersion) -> Regressor:
    model_uri = client.get_model_version_download_uri(
        model_version.name, model_version.version
    )
    return store.mlflow.load_model(model_uri).get_raw_model()


def tag_model_version(
    client, model_version: ModelVersion, tags: dict[str, str]
) -> None:
    if model_version.run_id:
        tags = {"run_id": model_version.run_id, **tags}
    for key, value in tags.items():
        client.set_model_version_tag(
            model_version.name,
            model_version.version,
            key,
            value,
        )


def calculate_test_rmse(model: Regressor, test_df: pd.DataFrame) -> float:
    return root_mean_squared_error(
        test_df[TARGET],
        model.predict(test_df[FEATURES]),  # type: ignore
    )


def promote_challenger(
    store,
    client,
    challenger: ModelVersion,
    challenger_model: Regressor,
    df: pd.DataFrame,
    challenger_rmse: float,
    test_window: dict[str, str],
    reason: str,
) -> ModelVersion:
    """Refit the challenger's recipe on all data and register it as the champion."""
    model_family = challenger.tags["model_family"]
    val_rmse = float(challenger.tags["val_rmse"])

    champion_model = clone(challenger_model) # type: ignore
    champion_model.fit(df[FEATURES], df[TARGET])

    model_info = log_selected_model(
        store, champion_model, CHAMPION_MODEL_NAME, model_family, df[FEATURES]
    )
    champion = register_model(
        store,
        model_info,
        CHAMPION_MODEL_NAME,
        CHAMPION_ALIAS,
        model_family,
        val_rmse,
    )
    tag_model_version(
        client,
        champion,
        {
            **test_window,
            "challenger_name": challenger.name,
            "challenger_version": challenger.version,
            "challenger_run_id": challenger.tags["run_id"],
            "expected_test_rmse": str(challenger_rmse),
            "promotion_decision": "promoted",
            "promotion_reason": reason,
        },
    )
    tag_model_version(
        client,
        challenger,
        {
            **test_window,
            "test_rmse": str(challenger_rmse),
            "champion_version": champion.version,
            "promotion_decision": "promoted",
            "promotion_reason": reason,
        },
    )
    print(
        f"Challenger v{challenger.version} refit on all data; "
        f"champion is v{champion.version}."
    )
    return champion


def compare_and_promote(
    store,
    client,
    challenger: ModelVersion,
    challenger_model: Regressor,
    champion: ModelVersion,
    challenger_rmse: float,
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_window: dict[str, str],
) -> ModelVersion | None:
    """Score the champion on the test window; promote the challenger only if it wins."""
    champion_rmse = calculate_test_rmse(load_model(store, client, champion), test_df)
    rmse_diff = challenger_rmse - champion_rmse
    rel_improvement = (champion_rmse - challenger_rmse) / champion_rmse
    promote = rel_improvement > MIN_REL_IMPROVEMENT

    print(f"Challenger v{challenger.version} test RMSE: {challenger_rmse:.4f}")
    print(f"Champion v{champion.version} test RMSE: {champion_rmse:.4f}")
    print(
        f"RMSE difference (challenger - champion): {rmse_diff:.4f} "
        f"({rel_improvement:+.1%} relative improvement)"
    )
    feast.mlflow.log_metrics(
        {
            "champion_test_rmse": champion_rmse,
            "rmse_diff": rmse_diff,
            "rel_improvement": rel_improvement,
        }
    )

    not_promoted_reason = (
        f"Challenger did not beat the champion by more than {MIN_REL_IMPROVEMENT:.1%}."
    )
    tag_model_version(
        client,
        champion,
        {
            **test_window,
            "test_rmse": str(champion_rmse),
            "promotion_decision": "demoted" if promote else "retained",
            "promotion_reason": (
                f"Demoted by challenger version {challenger.version}."
                if promote
                else f"Retained: {not_promoted_reason}"
            ),
        },
    )
    if not promote:
        tag_model_version(
            client,
            challenger,
            {
                **test_window,
                "test_rmse": str(challenger_rmse),
                "promotion_decision": "not_promoted",
                "promotion_reason": not_promoted_reason,
            },
        )
        print("Not promoted: champion retained.")
        return None

    new_champion = promote_challenger(
        store,
        client,
        challenger,
        challenger_model,
        df,
        challenger_rmse,
        test_window,
        reason=f"Promoted: challenger RMSE was {rel_improvement:.1%} lower than champion RMSE.",
    )
    client.set_registered_model_alias(
        champion.name, PREVIOUS_ALIAS, champion.version
    )
    return new_champion