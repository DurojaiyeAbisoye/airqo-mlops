import optuna
import pandas as pd
from feast import FeatureStore
from sklearn.model_selection import TimeSeriesSplit
from util import tune_cv
from xgboost import XGBRegressor


def xgboost_objective(
    trial: optuna.trial.Trial,
    store: FeatureStore,
    trainval_df: pd.DataFrame,
    cv: TimeSeriesSplit,
    features: list[str],
    target: str,
):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 30, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
    }
    model = XGBRegressor(**params, eval_metric="rmse")
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
