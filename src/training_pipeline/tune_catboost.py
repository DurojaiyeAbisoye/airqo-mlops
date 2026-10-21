import optuna
import pandas as pd
from catboost import CatBoostRegressor
from feast import FeatureStore
from sklearn.model_selection import TimeSeriesSplit
from util import tune_cv


def catboost_objective(
    trial: optuna.trial.Trial,
    store: FeatureStore,
    trainval_df: pd.DataFrame,
    cv: TimeSeriesSplit,
    features: list[str],
    target: str,
):
    params = {
        "iterations": trial.suggest_int("iterations", 300, 2000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 100.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
    }
    model = CatBoostRegressor(
        **params,
        eval_metric="RMSE",
        loss_function="RMSE",
        verbose=False,
    )
    X = trainval_df[features]
    y = trainval_df[target]

    return tune_cv(
        trial,
        store,
        cv,
        model,
        X,
        y,
    )
