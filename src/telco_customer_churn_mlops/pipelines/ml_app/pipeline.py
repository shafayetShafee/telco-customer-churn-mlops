"""
Kedro ML app pipeline
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes_ml.split_data import (
    train_calib_test_split_data
)
from .nodes_ml.hyperparameter_tuning import tune_xgb_pr_auc
from .nodes_ml.calibration import (
    calibrate_fitted_classifer,
    plot_calibration_comparison
)
from .nodes_ml.threshold_tuning import (
    tune_threshold,
    plot_threshold_metrics
)

from .nodes_ml.log_model import (
    log_calibrated_model
)

from .nodes_ml.model_registry import (
    evaluate_challenger_vs_champion,
    register_model_if_champion
)

# from .nodes_ml.model_train import (
#     fit_calibrated_final_model,
#     evaluate_model
# )

from .nodes_ml.predict import (
    infer_from_model
)

def create_ml_pipeline(**kwargs) -> Pipeline:
    training_pipeline = Pipeline(
        [
            Node(
                func=train_calib_test_split_data,
                inputs=[
                    "processed_train_data",
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
                func=calibrate_fitted_classifer,
                inputs=[
                    "optuna_best_model",
                    "X_calib", "y_calib",
                    "params:prefit_calib_options"
                ],
                outputs="calibrated_model",
                name="calibrate_fitted_classifier_node",
                tags=["training"]
            ),
            Node(
                func=tune_threshold,
                inputs=[
                    "calibrated_model",
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
                func=plot_calibration_comparison,
                inputs=[
                    "optuna_best_model",
                    "calibrated_model",
                    "X_test", "y_test"
                ],
                outputs="calibration_reliability_plot",
                name="plot_calibration_diagram_node",
                tags=["training"]
            ),
            # Node(
            #     func=evaluate_model,
            #     inputs=[
            #         "calibrated_model",
            #         "X_test", "y_test",
            #         "best_threshold"
            #     ],
            #     outputs="classification_report_df",
            #     name="model_evaluate_node",
            #     tags=["training"]
            # ),
            Node(
                func=log_calibrated_model,
                inputs=[
                    "X_test",
                    "calibrated_model",
                    "best_threshold"
                ],
                outputs=[
                    "calibrated_threshold_classifier",
                    "logged_model_info"
                ],
                name="calibrated_model_logging_node",
                tags=["training"]
            ),
            Node(
                func=evaluate_challenger_vs_champion,
                inputs=[
                    "logged_model_info",
                    "X_test", 
                    "y_test", 
                    "params:mlflow_evaluate_options"
                ],
                outputs="challenger_beats_champion",
                name="model_evaluation_node",
                tags=["training"]
            ),
            Node(
                func=register_model_if_champion,
                inputs=[
                    "logged_model_info",
                    "challenger_beats_champion",
                    "params:registry_options"
                ],
                outputs="registered_model_version",
                name="model_registration_node",
                tags=["training"]
            )
        ]
    )

    inference_pipeline = Pipeline(
        [
            Node(
                func=infer_from_model,
                inputs=[
                    "registered_model_version",
                    "processed_infer_data",
                    "params:inference_options"
                ],
                outputs="inference_result",
                name="infer_from_model_node",
                tags=["inference"] 
            )
        ]
    )

    return (
        training_pipeline 
+  inference_pipeline 
    )
