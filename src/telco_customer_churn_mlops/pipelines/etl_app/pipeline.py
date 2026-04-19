"""
This is a boilerplate pipeline 'etl_app'
generated using Kedro 1.2.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import (
    extract_encode_target_col,
    extract_preprocess_features_data
)


def create_etl_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=extract_encode_target_col,
                inputs=["telco", "params:target_column"],
                outputs="processed_target_col",
                name="target_column_processing_node",
                tags=["training"]
            ),
            Node(
                func=extract_preprocess_features_data,
                inputs=["telco", "params:features"],
                outputs="processed_features_data",
                tags=["training", "inference"]
            )
        ]
    )
