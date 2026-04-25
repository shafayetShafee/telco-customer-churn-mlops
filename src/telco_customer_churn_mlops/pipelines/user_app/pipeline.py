"""
This is a boilerplate pipeline 'user_app'
generated using Kedro 1.2.0
"""

from kedro.pipeline import Node, Pipeline  # noqa

from .nodes import serve_predictions


def create_user_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=serve_predictions,
                inputs=["telco_future"],
                outputs="served_predictions",
                name="serving_predictions_node",
                tags=["user"]
            )
        ]
    )
