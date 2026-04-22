import logging
import warnings

import mlflow
import numpy as np
import pandas as pd
from mapie.calibration import VennAbersCalibrator
from mlflow.exceptions import MlflowException
from mlflow.models import MetricThreshold

logger = logging.getLogger(__name__)

MODEL_NAME = "calibrated_full_xgb_model"
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"


def _get_champion_run_id() -> str | None:
    """
    Fetch the current champion's run_id from the MLflow model registry.
    Returns None if no champion is registered yet.
    """
    client = mlflow.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
        logger.info("Current champion: version %s (run: %s)", mv.version, mv.run_id)
        return mv.run_id
    except MlflowException:
        logger.info("No champion registered yet.")
        return None


def _get_champion_version() -> str | None:
    """
    Fetch the current champion's model version from the MLflow model registry.
    Returns None if no champion is registered yet.
    """
    client = mlflow.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
        return mv.version
    except MlflowException:
        return None


def _evaluate_on_predictions(
    model_preds: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target: str,
) -> mlflow.models.EvaluationResult:
    """
    Run mlflow.evaluate on static predictions — no model URI needed.

    Parameters
    ----------
    model_preds : np.ndarray
        Binary class predictions (0/1) from the model.
    X_test : pd.DataFrame
        Feature matrix for the test set.
    y_test : pd.Series
        True labels for the test set.
    target : str
        Name of the target column.

    Returns
    -------
    mlflow.models.EvaluationResult
    """
    eval_data = X_test.copy()
    eval_data[target] = y_test.values
    eval_data["prediction"] = model_preds

    return mlflow.models.evaluate(
        data=eval_data,
        targets=target,
        predictions="prediction",
        model_type="classifier",
    )


def evaluate_challenger_vs_champion(
    calibrated_model: VennAbersCalibrator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    registry_options: dict,
) -> bool:
    """
    Evaluate the challenger (held-out calibrated model) against the current
    champion on the test set, using static predictions to avoid needing a
    model URI.

    The held-out calibrated_model is used as an honest proxy for the full
    model's expected production performance — both are trained with the same
    hyperparameters and threshold, but the held-out model has never seen X_test.

    Parameters
    ----------
    calibrated_model : VennAbersCalibrator
        Held-out calibrated model fitted on X_train + X_calib only.
        Used as an honest performance proxy — never registered to production.
    X_test : pd.DataFrame
        Held-out test features — unseen by calibrated_model.
    y_test : pd.Series
        Held-out test labels.
    threshold : float
        Decision threshold applied to predict_proba output.
    registry_options : dict
        Registry configuration:
            - target (str): name of the target column
            - eval_metric (str): metric to compare on, default "recall_score"
            - min_absolute_change (float): minimum absolute improvement required, default 0.0
            - min_relative_change (float): minimum relative improvement required, default 0.0

    Returns
    -------
    bool
        True if challenger beats (or matches) the champion, or if no champion exists.
        False if the champion is still better.
    """
    warnings.filterwarnings("ignore")

    target = registry_options.get("target", "churn")
    eval_metric = registry_options.get("eval_metric", "recall_score")
    eval_metric_threshold = registry_options.get("eval_metric_threshold", 0.8)
    min_absolute_change = registry_options.get("min_absolute_change", 0.05)
    min_relative_change = registry_options.get("min_relative_change", 0.05)

    challenger_proba = calibrated_model.predict_proba(X_test)[:, 1]
    challenger_preds = (challenger_proba >= threshold).astype(int)
    challenger_result = _evaluate_on_predictions(challenger_preds, X_test, y_test, target)

    logger.info(
        "Challenger %s: %.4f",
        eval_metric,
        challenger_result.metrics[eval_metric],
    )

    champion_run_id = _get_champion_run_id()
    if not champion_run_id:
        logger.info("No champion found — challenger wins by default (first run).")
        return True

    # Champion predictions — load from registry and predict
    champion_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    champion_preds = champion_model.predict(X_test)
    champion_result = _evaluate_on_predictions(champion_preds, X_test, y_test, target)

    logger.info(
        "Champion   %s: %.4f",
        eval_metric,
        champion_result.metrics[eval_metric],
    )

    thresholds = {
        eval_metric: MetricThreshold(
            threshold=eval_metric_threshold,
            min_absolute_change=min_absolute_change,
            min_relative_change=min_relative_change,
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
    calibrated_full_xgb_model: VennAbersCalibrator,  # Kedro DAG dependency only
    challenger_beats_champion: bool,
    registry_options: dict,
) -> None:
    """
    Register the full model (already logged to MLflow in fit_calibrated_final_model)
    as champion if the challenger won evaluation or always_replace=True.

    The full model was already logged as a pyfunc artifact during fitting —
    this function only handles registry aliasing and tagging.

    Parameters
    ----------
    calibrated_full_xgb_model : VennAbersCalibrator
        Present only to declare the Kedro node dependency on fit_calibrated_final_model.
        Not used directly.
    challenger_beats_champion : bool
        Result from evaluate_challenger_vs_champion.
    registry_options : dict
        Registry configuration:
            - always_replace (bool): promote regardless of evaluation, default False
            - eval_metric (str): logged as a tag on the model version
    """
    always_replace = registry_options.get("always_replace", False)
    eval_metric = registry_options.get("eval_metric", "recall_score")

    if not (challenger_beats_champion or always_replace):
        logger.info(
            "Challenger did not beat champion and always_replace=False — skipping registration."
        )
        return

    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError(
            "register_model_if_champion must be called within an active MLflow run."
        )

    client = mlflow.MlflowClient()

    # Snapshot current champion version before we overwrite the alias
    current_champion_version = _get_champion_version()

    # Register the full model from the active run
    model_uri = f"runs:/{active_run.info.run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    # Tag the new version before aliasing — clean audit trail
    client.set_model_version_tag(MODEL_NAME, mv.version, "candidate_type", "champion")
    client.set_model_version_tag(MODEL_NAME, mv.version, "deployment_status", "production")
    client.set_model_version_tag(MODEL_NAME, mv.version, "eval_metric", eval_metric)
    client.set_model_version_tag(
        MODEL_NAME, mv.version, "promoted_by",
        "always_replace" if always_replace else "evaluation",
    )

    # Retire old champion tags before reassigning alias
    if current_champion_version:
        client.set_model_version_tag(
            MODEL_NAME, current_champion_version, "candidate_type", "retired_champion"
        )
        client.set_model_version_tag(
            MODEL_NAME, current_champion_version, "deployment_status", "retired"
        )

    # Promote — champion alias moves to new version
    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, mv.version)

    # Challenger alias points to this version until the next candidate arrives
    # (will be reassigned at the start of the next training run)
    client.set_registered_model_alias(MODEL_NAME, CHALLENGER_ALIAS, mv.version)

    logger.info(
        "Registered full model as champion: version %s (run: %s) | previous champion: %s",
        mv.version,
        active_run.info.run_id,
        current_champion_version or "none",
    )