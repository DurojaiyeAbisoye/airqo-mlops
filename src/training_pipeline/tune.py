from datetime import datetime, timezone

import feast.mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from feast import FeatureStore
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error
from tune_catboost import catboost_objective
from tune_lightgbm import lightgbm_objective
from tune_xgboost import xgboost_objective
from util import (
    FEATURE_SERVICE_NAME,
    FEATURES,
    TARGET,
    build_training_config,
    create_feature_store,
    load_entity_df,
    load_historical_data,
    make_cv,
    tune_objective,
)
from xgboost import XGBRegressor

MODEL_NAME = FEATURE_SERVICE_NAME
CV_TEST_SIZE = 30
CV_GAP = 0
CV_N_SPLITS = 5
LIGHTGBM_SKOPS_TRUSTED_TYPES = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMRegressor",
]

MODEL_FAMILY_CONFIG = {
    "xgboost": xgboost_objective,
    "catboost": catboost_objective,
    "lightgbm": lightgbm_objective,
}


def log_selected_model(
    store: FeatureStore,
    model: XGBRegressor | CatBoostRegressor | LGBMRegressor,
    model_family: str,
    features_df: pd.DataFrame,
) -> None:
    input_example = features_df.iloc[[0]]
    output_example = model.predict(input_example)
    signature = feast.mlflow.models.infer_signature(
        input_example,
        output_example,
    )

    if model_family == "catboost":
        store.mlflow.mlflow.catboost.log_model(
            model,
            name=MODEL_NAME,
            signature=signature,
        )
    elif model_family == "xgboost":
        feast.mlflow.xgboost.log_model(
            model,
            artifact_path=FEATURE_SERVICE_NAME,
            signature=signature,
        )
    else:
        feast.mlflow.lightgbm.log_model(
            model,
            artifact_path=FEATURE_SERVICE_NAME,
            signature=signature,
            skops_trusted_types=LIGHTGBM_SKOPS_TRUSTED_TYPES,
        )


def main() -> None:
    store = create_feature_store()
    entity_df = load_entity_df()
    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with store.mlflow.start_run(f"Training-{run_timestamp}"):
        df, trainval_df, test_df = load_historical_data(store, entity_df)
        cv = make_cv(
            test_size=CV_TEST_SIZE,
            gap=CV_GAP,
            n_splits=CV_N_SPLITS,
        )

        training_config = build_training_config(
            FEATURE_SERVICE_NAME,
            FEATURES,
            TARGET,
            df,
            trainval_df,
            test_df,
            cv,
            test_size=CV_TEST_SIZE,
            gap=CV_GAP,
        )
        feast.mlflow.log_dict(training_config, "training_config.json")

        studies = {}
        for model_family, model_objective_fn in MODEL_FAMILY_CONFIG.items():
            studies[model_family] = tune_objective(
                model_family,
                model_objective_fn,
                store,
                trainval_df,
                cv,
                FEATURES,
                TARGET,
            )

        best_model_family, best_study = min(
            studies.items(),
            key=lambda item: item[1].best_value,
        )
        best_params = best_study.best_params
        best_cv_val_rmse = best_study.best_value
        if best_model_family == "xgboost":
            selected_model = XGBRegressor(**best_params, eval_metric="rmse")
        elif best_model_family == "catboost":
            selected_model = CatBoostRegressor(
                **best_params,
                eval_metric="RMSE",
                loss_function="RMSE",
                verbose=False,
            )
        elif best_model_family == "lightgbm":
            selected_model = LGBMRegressor(
                **best_params,
                eval_metric="rmse",
                verbosity=0,
            )
        else:
            raise ValueError(f"Unknown model family: {best_model_family}")
        print(
            f"Selected {best_model_family} with mean CV validation RMSE: "
            f"{best_cv_val_rmse:.4f}"
        )

        with store.mlflow.start_run("Test-eval", nested=True):
            selected_model.fit(trainval_df[FEATURES], trainval_df[TARGET[0]])
            test_predictions = np.asarray(selected_model.predict(test_df[FEATURES]))
            test_rmse = root_mean_squared_error(
                test_df[TARGET[0]],
                test_predictions,
            )
            feast.mlflow.log_params({"selected_model_family": best_model_family})
            feast.mlflow.log_metrics(
                {
                    "best_cv_val_rmse": best_cv_val_rmse,
                    "test_rmse": test_rmse,
                }
            )
        print(f"Test RMSE for selected model: {test_rmse:.4f}")

        with store.mlflow.start_run("Model-Deployment", nested=True):
            selected_model.fit(df[FEATURES], df[TARGET[0]])
            feast.mlflow.log_params(
                {
                    "selected_model_family": best_model_family,
                    **{f"best_{name}": value for name, value in best_params.items()},
                }
            )
            feast.mlflow.log_metric("val_rmse", best_cv_val_rmse)
            log_selected_model(
                store,
                selected_model,
                best_model_family,
                df[FEATURES],
            )


if __name__ == "__main__":
    main()
