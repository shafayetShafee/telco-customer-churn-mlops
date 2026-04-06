"""
This is a boilerplate pipeline 'ml_app'
generated using Kedro 1.2.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes_ml.split_data import (
    train_calib_test_split_data
)
from .nodes_ml.hyperparameter_tuning import tune_xgb_pr_auc
from .nodes_ml.threshold_tuning import (
    tune_threshold,
    plot_threshold_metrics
)
from .nodes_ml.model_train import (
    train_xgb_model,
    evaluate_model
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=train_calib_test_split_data,
                inputs=[
                    "processed_features_data",
                    "processed_target_col",
                    "params:split_options"
                ],
                outputs=[
                    "X_train", "X_calib", "X_test",
                    "y_train", "y_calib", "y_test"
                ],
                name="data_split_node",
                tags=["training"]
            ),
            Node(
                func=tune_xgb_pr_auc,
                inputs=[
                    "X_train", "y_train",
                    "params:optuna_options",
                ],
                outputs=[
                    "optuna_best_model",
                    "optuna_best_params",
                    "optuna_search"
                ],
                name="hyperparameter_tuning_node",
                tags=["training"]
            ),
            Node(
                func=tune_threshold,
                inputs=[
                    "optuna_best_model",
                    "X_calib", "y_calib",
                    "params:min_recall"
                ],
                outputs=[
                    "tuning_threshold_df",
                    "best_threshold"
                ],
                name="threshold_tuning_node",
                tags=["training"]
            ),
            Node(
                func=plot_threshold_metrics,
                inputs=[
                    "tuning_threshold_df",
                    "best_threshold"
                ],
                outputs="threshold_metrics_plot",
                name="plot_threshold_metrics_node",
                tags=["training"]
            ),
            Node(
                func=train_xgb_model,
                inputs=[
                    "X_train", "y_train",
                    "optuna_best_params"
                ],
                outputs="xgb_model",
                name="train_xgb_model_node",
                tags=["training"]
            ),
            Node(
                func=evaluate_model,
                inputs=[
                    "xgb_model",
                    "X_test", "y_test",
                    "best_threshold"
                ],
                outputs="classification_report_df",
                name="model_evaluate_node",
                tags=["training"]
            )
        ]
    )
