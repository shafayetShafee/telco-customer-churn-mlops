import logging
import warnings

import mlflow
import pandas as pd
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.models import EvaluationResult, MetricThreshold
from mlflow.models.model import ModelInfo
from mlflow.pyfunc import PyFuncModel
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)

from telco_customer_churn_mlops.configs import (
    MlflowEvaluateConfig,
    ModelRegistryConfig,
)

logger = logging.getLogger(__name__)


def _get_champion_version(model_name: str, champion_alias: str) -> ModelVersion | None:
    """
    Fetch the current champion's model version from the MLflow model registry.
    Returns None if no champion is registered yet.

    Parameters
    ----------
    model_name : str
        The registered model name in MLflow.
    champion_alias : str
        The alias used to identify the champion version.
    """
    client = mlflow.MlflowClient()
    try:
        return client.get_model_version_by_alias(model_name, champion_alias)
    except MlflowException:
        logger.info("No champion registered yet.")
        return None


def _evaluate_on_predictions(
    model: PyFuncModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target: str,
) -> EvaluationResult:
    """
    Run mlflow.models.evaluate using a loaded PyFuncModel.

    Parameters
    ----------
    model : mlflow.pyfunc.PyFuncModel
        A loaded MLflow pyfunc model.

    X_test : pd.DataFrame
        Held-out feature matrix — unseen during training and calibration.

    y_test : pd.Series
        Held-out true labels.

    target : str
        Name of the target column used in the evaluation dataset.

    Returns
    -------
    mlflow.models.EvaluationResult
        Evaluation result containing metrics and artifacts.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        eval_data = X_test.copy()
        eval_data[target] = y_test.values

        prediction_proba = model.unwrap_python_model().predict_proba(X_test)[:, 1]

        mlflow.log_metrics(
            {
                "precision_recall_auc": average_precision_score(
                    y_true=y_test, y_score=prediction_proba
                ),
                "roc_auc": roc_auc_score(y_true=y_test, y_score=prediction_proba),
                "log_loss": log_loss(y_true=y_test, y_pred=prediction_proba),
            }
        )

        return mlflow.models.evaluate(
            model=model,
            data=eval_data,
            targets=target,
            model_type="classifier",
            evaluators="default",
            feature_names=list(X_test.columns),
            evaluator_config={
                "log_model_explainability": False
                # "log_explainer": False,
                # "explainer_type": "permutation",
                # "log_metrics_with_dataset_info": False
            },
        )


def evaluate_challenger_vs_champion(
    challenger_model_info: ModelInfo,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    mlflow_evaluate_options: dict,
    model_registry_options: dict,
) -> bool:
    """
    Evaluate the challenger (logged in the current MLflow run) against the
    current champion on the test set.

    Parameters
    ----------
    challenger_model_info : ModelInfo
        The mlflow.models.model.ModelInfo object from the logged model
        (i.e. challenger model) of the training pipeline.

    X_test : pd.DataFrame
        Held-out test features — unseen during training and calibration.

    y_test : pd.Series
        Held-out test labels.

    mlflow_evaluate_options : dict
        Evaluation configuration, see MlflowEvaluateConfig for fields.

    model_registry_options : dict
        Model registry configuration, see ModelRegistryConfig for fields.

    Returns
    -------
    bool
        True if the challenger beats (or matches) the champion, or if no
        champion exists (first run). False if the champion is still better.
    """
    eval_cfg = MlflowEvaluateConfig.from_params(mlflow_evaluate_options)
    reg_cfg = ModelRegistryConfig.from_params(model_registry_options)

    logger.info("Challenger model ID: %s", challenger_model_info.model_id)
    logger.info("Challenger model run ID: %s", challenger_model_info.run_id)

    challenger_model = mlflow.pyfunc.load_model(
        model_uri=challenger_model_info.model_uri, suppress_warnings=True
    )
    challenger_result = _evaluate_on_predictions(
        model=challenger_model, X_test=X_test, y_test=y_test, target=eval_cfg.target
    )

    logger.info(
        "Challenger %s: %.4f",
        eval_cfg.eval_metric,
        challenger_result.metrics[eval_cfg.eval_metric],
    )

    champion_version = _get_champion_version(reg_cfg.model_name, reg_cfg.champion_alias)
    if not champion_version:
        logger.info("No champion found — challenger wins by default (first run).")
        return True

    logger.info("Champion model run ID: %s", champion_version.run_id)

    champion_model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{reg_cfg.model_name}@{reg_cfg.champion_alias}",
        suppress_warnings=True,
    )
    champion_result = _evaluate_on_predictions(
        model=champion_model, X_test=X_test, y_test=y_test, target=eval_cfg.target
    )

    logger.info(
        "Champion %s: %.4f",
        eval_cfg.eval_metric,
        champion_result.metrics[eval_cfg.eval_metric],
    )

    thresholds = {
        eval_cfg.eval_metric: MetricThreshold(
            threshold=eval_cfg.eval_metric_threshold,
            min_absolute_change=eval_cfg.min_absolute_change,
            min_relative_change=eval_cfg.min_relative_change,
            greater_is_better=True,
        )
    }

    try:
        mlflow.validate_evaluation_results(
            validation_thresholds=thresholds,
            candidate_result=challenger_result,
            baseline_result=champion_result,
        )
        logger.info("Challenger beats champion — promoting to champion.")
        return True
    except MlflowException as e:
        logger.info("Challenger did not beat champion: %s", e)
        return False


def register_model_if_champion(
    challenger_model_info: ModelInfo,
    challenger_beats_champion: bool,
    registry_options: dict,
) -> ModelVersion:
    """
    Register a model artifact as champion if the challenger won evaluation
    or always_replace=True. Returns the champion ModelVersion object —
    either the newly registered version or the existing champion.

    Can be used for any MLflow-loggable artifact, such as an ML model or
    a SHAP explainer, as long as a corresponding registry configuration is
    provided.

    Parameters
    ----------
    challenger_model_info : ModelInfo
        The mlflow.models.model.ModelInfo object from the logged model
        (i.e. challenger model) of the training pipeline.

    challenger_beats_champion : bool
        Result from evaluate_challenger_vs_champion. If True, the challenger
        is promoted to champion.

    registry_options : dict
        Registry configuration, see ModelRegistryConfig for fields.

    Returns
    -------
    mlflow.entities.model_registry.ModelVersion
        The champion ModelVersion object — either the newly registered
        version if the challenger was promoted, or the existing champion
        if registration was skipped.

    Raises
    ------
    RuntimeError
        If called outside an active MLflow run, or if model registration fails.
    """
    reg_cfg = ModelRegistryConfig.from_params(registry_options)

    current_champion_version = _get_champion_version(
        reg_cfg.model_name, reg_cfg.champion_alias
    )

    if current_champion_version:
        if not (challenger_beats_champion or reg_cfg.always_replace):
            logger.info(
                "Challenger did not beat champion and always_replace=False — skipping registration."
            )
            return current_champion_version

    client = mlflow.MlflowClient()
    mv = mlflow.register_model(
        model_uri=challenger_model_info.model_uri, name=reg_cfg.model_name
    )

    client.set_model_version_tag(
        reg_cfg.model_name, mv.version, "candidate_type", "champion"
    )
    client.set_model_version_tag(
        reg_cfg.model_name, mv.version, "deployment_status", "production"
    )
    if reg_cfg.eval_metric:
        client.set_model_version_tag(
            reg_cfg.model_name, mv.version, "eval_metric", reg_cfg.eval_metric
        )

    if current_champion_version is None:
        promoted_by = "first_run"
    elif reg_cfg.always_replace:
        promoted_by = "always_replace"
    else:
        promoted_by = "evaluation"

    client.set_model_version_tag(
        reg_cfg.model_name, mv.version, "promoted_by", promoted_by
    )

    if current_champion_version:
        client.set_model_version_tag(
            reg_cfg.model_name,
            current_champion_version.version,
            "candidate_type",
            "retired_champion",
        )
        client.set_model_version_tag(
            reg_cfg.model_name,
            current_champion_version.version,
            "deployment_status",
            "retired",
        )

    client.set_registered_model_alias(
        reg_cfg.model_name, reg_cfg.champion_alias, mv.version
    )

    if current_champion_version:
        logger.info(
            "Retiring previous champion model — name: %s | version: %s | model_uri: %s\n"
            "(generated from run_id: %s )",
            current_champion_version.name,
            current_champion_version.version,
            current_champion_version.source,
            current_champion_version.run_id,
        )

    logger.info(
        "Current registered champion model — name: %s | version: %s | model_uri: %s\n"
        "(generated from run_id: %s | logged run_id: %s )",
        mv.name,
        mv.version,
        mv.source,
        mv.run_id,
        challenger_model_info.run_id,
    )

    return mv
