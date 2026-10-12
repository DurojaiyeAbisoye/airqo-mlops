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
from feast.types import Float32, Float64, String

REPO_ROOT = Path(__file__).resolve().parents[2]


feature_service_name = "pm25_regression"

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


entities = [
    Entity(name="country", value_type=ValueType.STRING),
    Entity(name="city", value_type=ValueType.STRING),
    Entity(name="street", value_type=ValueType.STRING),
]


air_quality_feature_view = FeatureView(
    name="air_quality_fv",
    source=air_quality_data_source,
    schema=[
        Field(name="pm25", dtype=Float64),
    ]
    + [
        Field(name="country", dtype=String),
        Field(name="city", dtype=String),
        Field(name="street", dtype=String),
        Field(name="pm25", dtype=Float64),
    ],
    entities=entities,
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
    ]
    + [
        Field(name="country", dtype=String),
        Field(name="city", dtype=String),
        Field(name="street", dtype=String),
        Field(name="pm25", dtype=Float64),
    ],
    entities=entities,
    offline=True,
    enable_validation=True,
)


pm25_feature_service = FeatureService(
    name=feature_service_name,
    features=[air_quality_feature_view, weather_feature_view],
)

store = FeatureStore(
    repo_path=f"{REPO_ROOT}/src/feature_repo",
    fs_yaml_file=Path(f"{REPO_ROOT}/src/feature_repo/feature_store.yaml"),
)


if __name__ == "__main__":
    store.apply(
        [
            *entities,
            air_quality_feature_view,
            weather_feature_view,
            pm25_feature_service,
        ]
    )
