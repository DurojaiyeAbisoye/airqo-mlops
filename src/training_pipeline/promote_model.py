import feast.mlflow
import pandas as pd
from mlflow.entities.model_registry import ModelVersion
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
from util import (
    FEATURES,
    REGISTERED_MODEL_NAME,
    TARGET,
    create_feature_store,
    load_entity_df,
    load_historical_data,
    log_selected_model,
    register_model,
)

CHALLENGER_ALIAS = "challenger"
CHAMPION_ALIAS = "champion"
PREVIOUS_ALIAS = "previous"


def get_model_version(client, alias: str) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
    except:
        if alias != CHAMPION_ALIAS:
            raise
        return None


def load_model(store, client, model_version: ModelVersion) -> XGBRegressor|CatBoostRegressor|LGBMRegressor:
    model_uri = client.get_model_version_download_uri(
        REGISTERED_MODEL_NAME, model_version.version
    )
    return store.mlflow.load_model(model_uri).get_raw_model()


def calculate_test_rmse(model: XGBRegressor|CatBoostRegressor|LGBMRegressor, test_df: pd.DataFrame) -> float:
    return root_mean_squared_error(
        test_df[TARGET],
        model.predict(test_df[FEATURES]), # type: ignore
    )


def tag_model_version(
    client, model_version: ModelVersion, tags: dict[str, str]
) -> None:
    if model_version.run_id:
        tags = {"run_id": model_version.run_id, **tags}
    for key, value in tags.items():
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            model_version.version,
            key,
            value,
        )


def initialize_champion(
    store,
    client,
    challenger: ModelVersion,
    challenger_model: XGBRegressor|CatBoostRegressor|LGBMRegressor,
    df: pd.DataFrame,
    test_rmse: float,
    val_rmse: float,
    test_window: dict[str, str],
) -> None:
    model_family = challenger.tags["model_family"]
    reason = "Promoted as the initial champion."

    feast.mlflow.log_metrics({"val_rmse": val_rmse, "test_rmse": test_rmse})
    challenger_model.fit(df[FEATURES], df[TARGET])
    model_info = log_selected_model(
        store,
        challenger_model,
        model_family,
        df[FEATURES],
    )
    champion = register_model(
        store,
        model_info,
        CHAMPION_ALIAS,
        model_family,
        val_rmse,
    )

    tags = {
        **test_window,
        "test_rmse": str(test_rmse),
        "promotion_decision": "promoted",
        "promotion_reason": reason,
    }
    tag_model_version(client, challenger, tags)
    tag_model_version(client, champion, tags)
    print(f"Challenger refit on all data and set as champion v{champion.version}.")


def compare_and_promote(
    store,
    client,
    challenger: ModelVersion,
    champion: ModelVersion,
    challenger_rmse: float,
    val_rmse: float,
    test_df: pd.DataFrame,
    test_window: dict[str, str],
) -> None:
    champion_rmse = calculate_test_rmse(load_model(store, client, champion), test_df)
    rmse_diff = challenger_rmse - champion_rmse
    promote = challenger_rmse < champion_rmse
    challenger_reason = (
        "Promoted: challenger RMSE was lower than champion RMSE."
        if promote
        else "Not promoted: challenger RMSE was not lower than champion RMSE."
    )
    champion_reason = (
        f"Demoted by challenger version {challenger.version}."
        if promote
        else "Retained: challenger RMSE was not lower than champion RMSE."
    )

    print(f"Challenger v{challenger.version} test RMSE: {challenger_rmse:.4f}")
    print(f"Champion v{champion.version} test RMSE: {champion_rmse:.4f}")
    print(f"RMSE difference (challenger - champion): {rmse_diff:.4f}")

    
    feast.mlflow.log_metrics(
        {
            "test_rmse": challenger_rmse,
            "challenger_test_rmse": challenger_rmse,
            "champion_test_rmse": champion_rmse,
            "rmse_diff": rmse_diff,
            "val_rmse": val_rmse,
        }
    )
    tag_model_version(
        client,
        challenger,
        {
            **test_window,
            "test_rmse": str(challenger_rmse),
            "promotion_decision": "promoted" if promote else "not_promoted",
            "promotion_reason": challenger_reason,
        },
    )
    tag_model_version(
        client,
        champion,
        {
            **test_window,
            "test_rmse": str(champion_rmse),
            "promotion_decision": "demoted" if promote else "retained",
            "promotion_reason": champion_reason,
        },
    )

    if promote:
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            PREVIOUS_ALIAS,
            champion.version,
        )
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            CHAMPION_ALIAS,
            challenger.version,
        )

    print(challenger_reason)


def main() -> None:
    store = create_feature_store()
    feast.mlflow.set_tracking_uri(store.mlflow._tracking_uri)
    
    with store.mlflow.start_run("Initialize Champion"):
        client = feast.mlflow.MlflowClient(store.mlflow._tracking_uri)
        challenger = get_model_version(client, CHALLENGER_ALIAS)
        if challenger is None:
            raise RuntimeError("The challenger alias does not exist.")

        df, _, test_df = load_historical_data(store, load_entity_df())
        test_window = {
            "test_window_start": test_df["event_timestamp"].min().isoformat(),
            "test_window_end": test_df["event_timestamp"].max().isoformat(),
        }
        challenger_model = load_model(store, client, challenger)
        challenger_rmse = calculate_test_rmse(challenger_model, test_df)
        val_rmse = float(challenger.tags["val_rmse"])
        champion = get_model_version(client, CHAMPION_ALIAS)

        if champion is None:
            initialize_champion(
                store,
                client,
                challenger,
                challenger_model,
                df,
                challenger_rmse,
                val_rmse,
                test_window,
            )
            return

        compare_and_promote(
            store,
            client,
            challenger,
            champion,
            challenger_rmse,
            val_rmse,
            test_df,
            test_window,
        )


if __name__ == "__main__":
    main()
