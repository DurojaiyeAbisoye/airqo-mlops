import optuna
import pandas as pd
from feast import FeatureStore
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from util import tune_cv


def lightgbm_objective(
    trial: optuna.trial.Trial,
    store: FeatureStore,
    trainval_df: pd.DataFrame,
    cv: TimeSeriesSplit,
    features: list[str],
    target: list[str],
):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
    }
    model = LGBMRegressor(**params, eval_metric="rmse", verbosity=0)
    X = trainval_df[features]
    y = trainval_df[target[0]]

    return tune_cv(
        trial,
        store,
        cv,
        model,
        X,
        y,
    )
