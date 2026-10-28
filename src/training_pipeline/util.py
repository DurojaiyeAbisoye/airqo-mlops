from datetime import datetime, timedelta, timezone
from pathlib import Path

import feast.mlflow
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from feast import FeatureStore
from lightgbm import LGBMRegressor
from mlflow.entities.logged_model import LoggedModel
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
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

TEST_WINDOW_DAYS = 14
MIN_REL_IMPROVEMENT = 0.0
N_TRIALS = 30

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
    return entity_df


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
        .sort_values("date")
        .reset_index(drop=True)
    )

    test_end_date = df.date.max()
    test_start_date = test_end_date - timedelta(days=TEST_WINDOW_DAYS)

    trainval_df = df[df.date <= test_start_date]
    test_df = df[df.date.between(test_start_date, test_end_date, inclusive="right")]
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
            "trainval_start": df["date"].min().isoformat(),
            "trainval_end": trainval_df["date"].max().isoformat(),
            "test_start": test_df["date"].min().isoformat(),
            "test_end": test_df["date"].max().isoformat(),
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
        feast.mlflow.log_params(model.get_params())

        folds = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            Xtrain, Xval = X.iloc[train_idx], X.iloc[val_idx]
            ytrain, yval = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(Xtrain, ytrain)
            ytrain_pred = np.asarray(model.predict(Xtrain))
            yval_pred = np.asarray(model.predict(Xval))

            scores = {
                "Train RMSE": root_mean_squared_error(ytrain, ytrain_pred),
                "Train R2": r2_score(ytrain, ytrain_pred),
                "Val RMSE": root_mean_squared_error(yval, yval_pred),
                "Val R2": r2_score(yval, yval_pred),
            }
            folds.append(scores)
            feast.mlflow.log_metrics(scores, step=fold)
            print(
                f"Fold {fold + 1}/{cv.get_n_splits()} | "
                + " | ".join(f"{k}: {v:.4f}" for k, v in scores.items())
            )

            trial.report(scores["Val RMSE"], step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        results = pd.DataFrame(folds)
        means, stds = results.mean(), results.std()
        feast.mlflow.log_metrics(
            {f"Mean {k}": v for k, v in means.items()}
            | {"Std Val RMSE": stds["Val RMSE"]}
        )
        feast.mlflow.log_dict(results.to_dict(orient="list"), "fold_scores.json")
        print("CV mean | " + " | ".join(f"{k}: {v:.4f}" for k, v in means.items()))
        return float(means["Val RMSE"])


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
            n_trials=N_TRIALS,
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
) -> LoggedModel:
    input_example = features_df.iloc[[0]]
    signature = feast.mlflow.models.infer_signature(
        input_example, model.predict(input_example)
    )

    if model_family == "catboost":
        feast.mlflow.catboost.log_model(model, name=model_name, signature=signature)
        store.mlflow._log_required_features()
    elif model_family == "xgboost":
        store.mlflow.log_model(model, model_name, flavor="xgboost", signature=signature)
    else:
        store.mlflow.log_model(
            model,
            model_name,
            flavor="lightgbm",
            signature=signature,
            skops_trusted_types=LIGHTGBM_SKOPS_TRUSTED_TYPES,
        )

    logged_model = feast.mlflow.last_logged_model()
    if logged_model is None:
        raise RuntimeError(f"No logged model found after logging {model_name}.")
    return logged_model


def register_model(
    store: FeatureStore,
    model_name: str,
    model_uri: str,
    alias: str,
    model_family: str,
    val_rmse: float,
) -> ModelVersion:
    model_version = store.mlflow.register_model(model_uri, name=model_name)
    print("Model Registered!")

    mlflow_client = feast.mlflow.MlflowClient(store.mlflow._tracking_uri)
    tags = {
        "model_family": model_family,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "val_rmse": str(val_rmse),
        "feast.feature_service": FEATURE_SERVICE_NAME,
    }
    run_id = store.mlflow.active_run_id
    if run_id:
        tags["run_id"] = run_id
    for key, value in tags.items():
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
            str(value),
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

    champion_model = clone(challenger_model)  # type: ignore
    champion_model.fit(df[FEATURES], df[TARGET])

    champion_logged_model = log_selected_model(
        store, champion_model, CHAMPION_MODEL_NAME, model_family, df[FEATURES]
    )
    champion = register_model(
        store,
        CHAMPION_MODEL_NAME,
        champion_logged_model.model_uri,
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
            "challenger_test_rmse": str(challenger_rmse),
            "promotion_decision": "promoted",
            "promotion_reason": reason,
        },
    )
    tag_model_version(
        client,
        challenger,
        {
            **test_window,
            "challenger_test_rmse": str(challenger_rmse),
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
    scores = {
        **test_window,
        "champion_test_rmse": str(champion_rmse),
        "challenger_test_rmse": str(challenger_rmse),
    }
    tag_model_version(
        client,
        champion,
        {
            "last_gate_window_start": test_window["test_window_start"],
            "last_gate_window_end": test_window["test_window_end"],
            "last_gate_champion_rmse": str(champion_rmse),
            "last_gate_challenger_rmse": str(challenger_rmse),
            "last_gate_challenger_version": challenger.version,
            "last_gate_decision": "demoted" if promote else "retained",
            "last_gate_reason": (
                f"Demoted by challenger version {challenger.version}."
                if promote
                else f"Retained: {not_promoted_reason}"
            ),
            "champion_test_rmse": str(champion_rmse),
            "rmse_diff": str(rmse_diff),
            "rel_improvement": str(rel_improvement),
        },
    )
    if not promote:
        tag_model_version(
            client,
            challenger,
            {
                **scores,
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
    client.set_registered_model_alias(champion.name, PREVIOUS_ALIAS, champion.version)
    return new_champion
