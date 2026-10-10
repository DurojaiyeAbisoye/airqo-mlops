import warnings
from datetime import timedelta
from pathlib import Path
import feast.mlflow
import pandas as pd
from feast import FeatureStore
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
warnings.filterwarnings("ignore")

feature_service_name = "pm25_regression"
# initialize the feature store
store = FeatureStore(
    repo_path="feature_repo", fs_yaml_file=Path("feature_repo/feature_store.yaml")
)
feast.mlflow.init(store)

entity_columns = ["city", "country", "state", "street", "date", "created_at"]
entity_df = pd.read_parquet("feature_repo/data/cleaned-air-quality-data.parquet")[
    entity_columns
]
entity_df = entity_df.rename(columns={"date": "event_timestamp"})

historical_df = (
    store.get_historical_features(
        entity_df=entity_df,
        features=store.get_feature_service(feature_service_name),
        filter_by_created_timestamp=True,
    )
    .to_df()
    .sort_values(by="event_timestamp")
    .reset_index(drop=True)
)

# split data
last_date = historical_df["event_timestamp"].max()
val_end_date = last_date - timedelta(days=7)
train_end_date = val_end_date - timedelta(days=7)


print(f"{historical_df.shape = }")
train_df = historical_df[historical_df.event_timestamp <= train_end_date]
val_df = historical_df[
    (historical_df.event_timestamp > train_end_date)
    & (historical_df.event_timestamp <= val_end_date)
]
test_df = historical_df[historical_df.event_timestamp > val_end_date]

print(train_df.shape, val_df.shape, test_df.shape)

# instantiate models


feast.mlflow.sklearn.autolog()
feast.mlflow.xgboost.autolog()
feast.mlflow.lightgbm.autolog()

rf = RandomForestRegressor(random_state=23)
hist = HistGradientBoostingRegressor(random_state=23)
xgb = XGBRegressor(random_state=23)
lgb = LGBMRegressor(random_state=25)
# define features and targets
target = ["pm25"]
features = train_df.drop(
    columns=["city", "country", "state", "street", "event_timestamp", "created_at"]
    + target
).columns.tolist()
print(f"{features = }")
print(f"{target = }")


def train_and_evaluate_model(model):

    with store.mlflow.start_run():
        Xtrain = train_df[features]
        Xval = val_df[features]
        # Xtest = test_df[features]

        ytrain = train_df[target].ffill()
        yval = val_df[target].ffill()
        # ytest = test_df[target]

        model.fit(Xtrain, ytrain)
        print(f"Fit {model.__class__.__name__}!")

        train_rmse = root_mean_squared_error(ytrain, model.predict(Xtrain))
        val_rmse = root_mean_squared_error(yval, model.predict(Xval))

        train_r2 = r2_score(ytrain, model.predict(Xtrain))
        val_r2 = r2_score(yval, model.predict(Xval))

        print(f"\tRMSE: \n\t\tTrain -> {train_rmse: .2f}\n\t\tVal -> {val_rmse: .2f}")
        print(f"\tR2: \n\t\tTrain -> {train_r2: .2f}\n\t\tVal -> {val_r2: .2f}")
        print()

        feast.mlflow.log_metrics(
            {
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "train_r2": train_r2,
                "val_r2": val_r2,
            }
        )
        store.mlflow.log_model(
            model,
            artifact_path=feature_service_name,
            input_example=Xtrain,
            skops_trusted_types=[
                "sklearn.tree._tree.Tree",
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBRegressor",
            ],
        )


if __name__ == "__main__":
    train_and_evaluate_model(xgb)
    train_and_evaluate_model(lgb)

    train_and_evaluate_model(rf)
    train_and_evaluate_model(hist)
