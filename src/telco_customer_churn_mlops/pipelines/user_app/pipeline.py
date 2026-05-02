from kedro.pipeline import Node, Pipeline  # noqa

from .nodes import serve_predictions, serve_shap_values
from telco_customer_churn_mlops.expectations import validate_dataset


def create_user_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=validate_dataset,
                inputs=[
                    "telco_future",
                    "params:features",
                    "params:column_types",
                    "params:infer_data_type",
                    "params:infer_data_type",
                ],
                outputs="gx_infer_result",
                name="gx_infer_validation_node",
                tags=["user"],
            ),
            Node(
                func=serve_predictions,
                inputs=["telco_future", "params:model_registry_options"],
                outputs="served_predictions",
                name="serving_predictions_node",
                tags=["user"],
            ),
            Node(
                func=serve_shap_values,
                inputs=[
                    "served_predictions",
                    "params:inference_options",
                    "params:explainer_registry_options",
                    "params:shap_options",
                ],
                outputs=[
                    "served_shap_values",
                    "inference_shap_barplot",
                    "inference_shap_bees_plot",
                ],
                name="serving_shap_values_node",
                tags=["user"],
            ),
        ]
    )
