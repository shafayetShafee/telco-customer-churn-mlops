"""
This is a boilerplate pipeline 'ml_app'
generated using Kedro 1.2.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes_ml.split_data import train_test_split_data
from .nodes_ml.hyperparameter_tuning import tune_xgb_pr_auc
from .nodes_ml.threshold_tuning import (
    tune_threshold,
    plot_threshold_metrics
)


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
            Node(
                func=tune_xgb_pr_auc,
                inputs=[
                    "X_train", "y_train",
                    "X_test", "y_test",
                    "params:optuna_n_trials",
                    "params:project_seed"
                ],
                outputs=[
                    "optuna_best_model",
                    "optuna_best_params",
                    "optuna_study"
                ],
                name="hyperparameter_tuning_node",
                tags=["training"]
            ),
            Node(
                func=tune_threshold,
                inputs=[
                    "optuna_best_model",
                    "X_train", "y_train",
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
            )
        ]
    )
