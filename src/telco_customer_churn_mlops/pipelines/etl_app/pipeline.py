"""
This is a boilerplate pipeline 'etl_app'
generated using Kedro 1.2.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import (
    extract_encode_target_col,
    extract_preprocess_features_data,
    fit_multi_cat_encoder,
    apply_multi_cat_encoder,
    clean_names
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
                outputs="cleaned_features_data",
                tags=["training", "inference"]
            ),
            Node(
                func=fit_multi_cat_encoder,
                inputs=[
                    "cleaned_features_data",
                    "params:multi_cat_features"
                ],
                outputs="multi_cat_encoder",
                tags=["training"]
            ),
            Node(
                func=apply_multi_cat_encoder,
                inputs=[
                    "cleaned_features_data", 
                    "multi_cat_encoder",
                    "params:multi_cat_features"
                ],
                outputs="multi_cat_encoded_df",
                tags=["training", "inference"]
            ),
            Node(
                func=clean_names,
                inputs="multi_cat_encoded_df",
                outputs="processed_features_data",
                tags=["training", "inference"]
            )
        ]
    )
