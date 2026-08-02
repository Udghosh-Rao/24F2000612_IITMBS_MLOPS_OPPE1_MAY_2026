import os
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64


DATA_FILE = os.environ.get(
    "FEAST_DATA_FILE",
    "data/iteration2.parquet"
)

stock = Entity(
    name="stock_name",
    join_keys=["stock_name"],
    value_type=ValueType.STRING,
)

stock_source = FileSource(
    name="stock_source",
    path=DATA_FILE,
    timestamp_field="timestamp",
)

stock_features = FeatureView(
    name="stock_features",
    entities=[stock],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="rolling_avg_10", dtype=Float64),
        Field(name="volume_sum_10", dtype=Float64),
    ],
    source=stock_source,
)
