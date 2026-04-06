"""
This is a boilerplate pipeline 'ml_app'
generated using Kedro 1.2.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes_ml.split_data import train_test_split_data
# from .nodes_ml.threshold_tuning import tune_threshold
from .nodes_ml.hyperparameter_tuning import tune_xgb_pr_auc


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=train_test_split_data,
                inputs=[
                    "processed_features_data", 
                    "processed_target_col",
                    "params:split_options"
                ],
                outputs=[
                    "X_train", "X_test",
                    "y_train", "y_test"
                ],
                tags=["training"]
            ),
            # Node(
            #     func=tune_xgb_pr_auc,
            #     inputs=[
            #         "X_train", "y_train",
            #         "X_test", "y_test"
            #     ],
            #     outputs=[
            #         "optuna_best_model",
            #         "optuna_best_params",
            #         "optuna_study"
            #     ]
            # )
        ]
    )
