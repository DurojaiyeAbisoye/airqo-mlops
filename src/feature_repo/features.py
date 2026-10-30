from pathlib import Path

from feast import (
    Entity,
    FeatureService,
    FeatureStore,
    FeatureView,
    Field,
    FileSource,
    ValueType,
)
from feast.data_format import ParquetFormat
from feast.types import Float32, Float64

REPO_ROOT = Path(__file__).resolve().parents[2]


PROJECT_NAME = "pm25_regression"
FEATURE_SERVICE_VERSION = 1
FEATURE_SERVICE_NAME = f"{PROJECT_NAME}_v{FEATURE_SERVICE_VERSION}"

country_entity = Entity(
    name="country",
    join_keys=["country"],
    value_type=ValueType.STRING,
)

city_entity = Entity(
    name="city",
    join_keys=["city"],
    value_type=ValueType.STRING,
)

street_entity = Entity(
    name="street",
    join_keys=["street"],
    value_type=ValueType.STRING,
)

days_before_forecast_entiy = Entity(
    name="days_before_forecast_day",
    join_keys=["days_before_forecast_day"],
    value_type=ValueType.STRING,
)

air_quality_data_source = FileSource(
    path=f"{REPO_ROOT}/src/feature_repo/data/cleaned_historical_sensor_data.parquet",
    name="air-quality_data",
    timestamp_field="date",
    created_timestamp_column="created_at",
    file_format=ParquetFormat(),
)

weather_data_source = FileSource(
    path=f"{REPO_ROOT}/src/feature_repo/data/weather_historical.parquet",
    name="weather_data",
    timestamp_field="date",
    created_timestamp_column="created_at",
    file_format=ParquetFormat(),
)

monitoring_data_source = FileSource(
    path=f"{REPO_ROOT}/src/feature_repo/data/predictions.parquet",
    name="monitoring_data",
    timestamp_field="date",
    file_format=ParquetFormat(),
)

air_quality_feature_view = FeatureView(
    name="air_quality_fv",
    source=air_quality_data_source,
    schema=[
        Field(name="pm25", dtype=Float64),
    ],
    entities=[country_entity, city_entity, street_entity],
    offline=True,
    enable_validation=True,
)

weather_feature_view = FeatureView(
    name="weather_fv",
    source=weather_data_source,
    schema=[
        Field(name=i, dtype=Float32)
        for i in [
            "temperature_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max",
            "wind_direction_10m_dominant",
        ]
    ],
    entities=[country_entity, city_entity, street_entity],
    offline=True,
    enable_validation=True,
)


monitoring_feature_view = FeatureView(
    name="monitoring_fv",
    source=monitoring_data_source,
    schema=[
        Field(name=i, dtype=Float32)
        for i in [
            "temperature_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max",
            "wind_direction_10m_dominant",
        ]
    ]
    + [
        Field(name="pm25", dtype=Float64),
    ]
    + [Field(name="predicted_pm25", dtype=Float64)],
    entities=[country_entity, city_entity, street_entity, days_before_forecast_entiy],
    offline=True,
    enable_validation=True,
)


pm25_feature_service = FeatureService(
    name=PROJECT_NAME,
    features=[air_quality_feature_view, weather_feature_view],
)

pm25_feature_service_v1 = FeatureService(
    name=FEATURE_SERVICE_NAME,
    features=[air_quality_feature_view, weather_feature_view],
)


store = FeatureStore(
    repo_path=f"{REPO_ROOT}/src/feature_repo",
    fs_yaml_file=Path(f"{REPO_ROOT}/src/feature_repo/feature_store.yaml"),
)
