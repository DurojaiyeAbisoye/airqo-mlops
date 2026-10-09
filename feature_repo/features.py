from datetime import timedelta
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
from feast.types import Float32

feature_service_name = "pm25_regression"

air_quality_data_source = FileSource(
    path="feature_repo/data/cleaned-air-quality-data.parquet",
    name="air-quality_data",
    timestamp_field="date",
    created_timestamp_column="created_at",
    file_format=ParquetFormat(),
)

weather_data_source = FileSource(
    path="feature_repo/data/weather-data.parquet",
    name="weather_data",
    timestamp_field="date",
    created_timestamp_column="created_at",
    file_format=ParquetFormat(),
)


air_quality_entities = weather_entities = [
    Entity(name="country", value_type=ValueType.STRING),
    Entity(name="city", value_type=ValueType.STRING),
    Entity(name="state", value_type=ValueType.STRING),
    Entity(name="street", value_type=ValueType.STRING),
]


air_quality_feature_view = FeatureView(
    name="air_quality_fv",
    source=air_quality_data_source,
    schema=[
        Field(name="pm25", dtype=Float32),
    ],
    entities=air_quality_entities,
    ttl=timedelta(days=7),
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
    entities=weather_entities,
    ttl=timedelta(days=7),
    offline=True,
    enable_validation=True,
)


pm25_feature_service = FeatureService(
    name=feature_service_name,
    features=[air_quality_feature_view, weather_feature_view],
)

store = FeatureStore(
    repo_path="feature_repo", fs_yaml_file=Path("feature_repo/feature_store.yaml")
)


if __name__ == "__main__":
    store.apply(
        [
            *air_quality_entities,
            air_quality_feature_view,
            weather_feature_view,
            pm25_feature_service,
        ]
    )
