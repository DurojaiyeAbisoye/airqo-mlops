from datetime import datetime, timezone

import feast.mlflow
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from tune_catboost import catboost_objective
from tune_lightgbm import lightgbm_objective
from tune_xgboost import xgboost_objective
from util import (
    CHALLENGER_ALIAS,
    CHALLENGER_MODEL_NAME,
    CHAMPION_ALIAS,
    CHAMPION_MODEL_NAME,
    FEATURE_SERVICE_NAME,
    FEATURES,
    TARGET,
    build_training_config,
    calculate_test_rmse,
    compare_and_promote,
    create_feature_store,
    get_model_version,
    load_entity_df,
    load_historical_data,
    log_selected_model,
    make_cv,
    promote_challenger,
    register_model,
    tune_objective,
)
from xgboost import XGBRegressor

CV_TEST_SIZE = 30
CV_GAP = 0
CV_N_SPLITS = 5

MODEL_FAMILY_CONFIG = {
    "xgboost": xgboost_objective,
    "catboost": catboost_objective,
    "lightgbm": lightgbm_objective,
}


def main() -> None:
    store = create_feature_store()
    entity_df = load_entity_df()
    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    client = feast.mlflow.MlflowClient(store.mlflow._tracking_uri)

    with store.mlflow.start_run(f"Training-{run_timestamp}"):
        df, trainval_df, test_df = load_historical_data(store, entity_df)
        store.mlflow.log_training_dataset(df, "full_df")
        store.mlflow.log_training_dataset(trainval_df, "trainval_df")
        test_window = {
            "test_window_start": test_df["event_timestamp"].min().isoformat(),
            "test_window_end": test_df["event_timestamp"].max().isoformat(),
        }
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
            selected_model = LGBMRegressor(**best_params, verbosity=-1)
        else:
            raise ValueError(f"Unknown model family: {best_model_family}")
        print(
            f"Selected {best_model_family} with mean CV validation RMSE: "
            f"{best_cv_val_rmse:.4f}"
        )

        # The challenger is fit on train+val only, so the test window stays unseen.
        selected_model.fit(trainval_df[FEATURES], trainval_df[TARGET])
        challenger_rmse = calculate_test_rmse(selected_model, test_df=test_df)
        feast.mlflow.log_dict(
            {"selected_model_family": best_model_family}, "model_family.json"
        )
        feast.mlflow.log_params(best_params)
        feast.mlflow.log_metrics(
            {"best_cv_val_rmse": best_cv_val_rmse, "test_rmse": challenger_rmse}
        )

        challenger_logged_model = log_selected_model(
            store,
            selected_model,
            CHALLENGER_MODEL_NAME,
            best_model_family,
            df[FEATURES],
        )
        challenger = register_model(
            store,
            CHALLENGER_MODEL_NAME,
            challenger_logged_model.model_uri,
            CHALLENGER_ALIAS,
            best_model_family,
            best_cv_val_rmse,
        )

        # Promotion Gate: runs inside the Training run so the Feast UI links both models to it.
        champion = get_model_version(client, CHAMPION_MODEL_NAME, CHAMPION_ALIAS)
        if champion is None:
            promote_challenger(
                store,
                client,
                challenger,
                selected_model,
                df,
                challenger_rmse,
                test_window,
                reason="Promoted as the initial champion.",
            )
        else:
            compare_and_promote(
                store,
                client,
                challenger,
                selected_model,
                champion,
                challenger_rmse,
                df,
                test_df,
                test_window,
            )


if __name__ == "__main__":
    main()
