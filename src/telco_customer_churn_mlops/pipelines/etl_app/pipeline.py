"""
Kedro pipeline `etl` that process both the training data and
inference data
"""

from kedro.pipeline import Node, Pipeline  # noqa

from .nodes import (
    extract_encode_target_col,
    extract_preprocess_features_data,
    fit_multi_cat_encoder,
    apply_multi_cat_encoder,
    clean_names,
)


def create_etl_pipeline(**kwargs) -> Pipeline:

    etl_train_data_pipeline = Pipeline(
        [
            Node(
                func=extract_encode_target_col,
                inputs=["telco", "params:target_column"],
                outputs="processed_target_col",
                name="target_column_processing_node",
                tags=["training"],
            ),
            Node(
                func=extract_preprocess_features_data,
                inputs=["telco", "params:features"],
                outputs="cleaned_features_data",
                name="features_data_preprocessing_node",
                tags=["training"],
            ),
            Node(
                func=fit_multi_cat_encoder,
                inputs=["cleaned_features_data", "params:multi_cat_features"],
                outputs="multi_cat_encoder",
                name="features_df_OHE_fitting_node",
                tags=["training"],
            ),
            Node(
                func=apply_multi_cat_encoder,
                inputs=[
                    "cleaned_features_data",
                    "multi_cat_encoder",
                    "params:multi_cat_features",
                ],
                outputs="multi_cat_encoded_df",
                name="features_df_OHE_transform_node",
                tags=["training"],
            ),
            Node(
                func=clean_names,
                inputs="multi_cat_encoded_df",
                outputs="processed_features_data",
                name="features_df_clean_names_node",
                tags=["training"],
            ),
        ]
    )

    etl_infer_data_pipeline = Pipeline(
        [
            Node(
                func=extract_preprocess_features_data,
                inputs=["telco_future", "params:features"],
                outputs="cleaned_inference_data",
                name="inference_data_preprocessing_node",
                tags=["inference"],
            ),
            Node(
                func=apply_multi_cat_encoder,
                inputs=[
                    "cleaned_inference_data",
                    "multi_cat_encoder",
                    "params:multi_cat_features",
                ],
                outputs="multi_cat_encoded_inference_df",
                name="inference_df_OHE_transform_node",
                tags=["inference"],
            ),
            Node(
                func=clean_names,
                inputs="multi_cat_encoded_inference_df",
                outputs="processed_inference_data",
                name="inference_df_clean_names_node",
                tags=["inference"],
            ),
        ]
    )

    return etl_train_data_pipeline + etl_infer_data_pipeline
