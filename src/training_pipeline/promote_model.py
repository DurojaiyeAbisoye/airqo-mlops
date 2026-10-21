import feast.mlflow
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode
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


def main() -> None:
    store = create_feature_store()
    client = feast.mlflow.MlflowClient(store.mlflow._tracking_uri)
    challenger = client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME, CHALLENGER_ALIAS
    )
    model_family = challenger.tags["model_family"]
    val_rmse = float(challenger.tags["val_rmse"])

    df, _, test_df = load_historical_data(store, load_entity_df())
    try:
        champion = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME, CHAMPION_ALIAS
        )
    except MlflowException as error:
        if error.error_code != ErrorCode.Name(ErrorCode.RESOURCE_DOES_NOT_EXIST):
            raise

        challenger_uri = client.get_model_version_download_uri(
            REGISTERED_MODEL_NAME, challenger.version
        )
        model = store.mlflow.load_model(challenger_uri).get_raw_model()
        model.fit(df[FEATURES], df[TARGET])
        with store.mlflow.start_run("Initialize Champion"):
            model_info = log_selected_model(store, model, model_family, df[FEATURES])
            champion = register_model(
                store, model_info, CHAMPION_ALIAS, model_family, val_rmse
            )
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            champion.version,
            "promotion_reason",
            "Promoted as the initial champion; refit on all available data.",
        )
        print(f"Challenger refit on all data and set as champion v{champion.version}.")
        return

    challenger_uri = client.get_model_version_download_uri(
        REGISTERED_MODEL_NAME, challenger.version
    )
    challenger_model = store.mlflow.load_model(challenger_uri).get_raw_model()
    champion_uri = client.get_model_version_download_uri(
        REGISTERED_MODEL_NAME, champion.version
    )
    champion_model = store.mlflow.load_model(champion_uri).get_raw_model()

    challenger_rmse = root_mean_squared_error(
        test_df[TARGET], challenger_model.predict(test_df[FEATURES])
    )
    champion_rmse = root_mean_squared_error(
        test_df[TARGET], champion_model.predict(test_df[FEATURES])
    )
    rmse_diff = challenger_rmse - champion_rmse
    test_start = test_df["event_timestamp"].min().isoformat()
    test_end = test_df["event_timestamp"].max().isoformat()
    promote = challenger_rmse < champion_rmse
    reason = (
        "Promoted: challenger RMSE was lower than champion RMSE."
        if promote
        else "Not promoted: challenger RMSE was not lower than champion RMSE."
    )

    print(f"Challenger v{challenger.version} test RMSE: {challenger_rmse:.4f}")
    print(f"Champion v{champion.version} test RMSE: {champion_rmse:.4f}")
    print(f"RMSE difference (challenger - champion): {rmse_diff:.4f}")
    for key, value in {
        "challenger_test_rmse": str(challenger_rmse),
        "champion_test_rmse": str(champion_rmse),
        "rmse_diff": str(rmse_diff),
        "test_window_start": test_start,
        "test_window_end": test_end,
        "promotion_decision": "promoted" if promote else "not_promoted",
        "promotion_reason": reason,
    }.items():
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME, challenger.version, key, value
        )

    if not promote:
        print(reason)
        return

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        champion.version,
        "demotion_reason",
        f"Replaced by challenger version {challenger.version}.",
    )
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, PREVIOUS_ALIAS, champion.version
    )
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS, challenger.version
    )
    print(reason)


if __name__ == "__main__":
    main()
