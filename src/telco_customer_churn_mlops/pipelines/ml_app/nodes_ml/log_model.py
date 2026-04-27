import logging

import mlflow
import numpy as np
import pandas as pd
from mapie.calibration import VennAbersCalibrator
from mlflow.models import infer_signature

from telco_customer_churn_mlops.configs import ModelRegistryConfig
from .utils import _ensure_fitted

logger = logging.getLogger(__name__)


class ThresholdClassifier(mlflow.pyfunc.PythonModel):
    """
    A custom MLflow PythonModel that wraps a calibrated classifier with a
    configurable decision threshold.

    By default, probabilistic classifiers use 0.5 as the decision boundary.
    This wrapper allows overriding that threshold, which is useful when
    optimising for precision, recall, or F1 on imbalanced datasets.

    The model is compatible with MLflow's pyfunc flavour and can be logged,
    registered, and served via the MLflow Model Registry.

    Parameters
    ----------
    calibrator : VennAbersCalibrator
        A fitted Venn-Abers calibrated classifier that exposes a
        `predict_proba(X)` method returning an array of shape
        (n_samples, n_classes).

    threshold : float
        Decision threshold in the range [0.0, 1.0]. Samples with predicted
        positive-class probability >= threshold are assigned class 1,
        otherwise class 0.

    """
    def __init__(self, calibrator: VennAbersCalibrator, threshold: float) -> None:
        _ensure_fitted(calibrator)
        self.calibrator = calibrator
        self.threshold = threshold

    def predict(self, model_input) -> np.ndarray:
        """
        Generate binary class predictions using the calibrated probabilities
        and the configured decision threshold.

        Parameters
        ----------
        model_input : pd.DataFrame | np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Binary predictions of shape (n_samples,), where 1 indicates the
            positive class and 0 the negative class.
        """
        proba = self.calibrator.predict_proba(model_input)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, model_input: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Return class probabilities as a standard sklearn-style array.

        Parameters
        ----------
        model_input : pd.DataFrame | np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, 2), where column 0 is the negative-class
            probability and column 1 is the positive-class probability.
        """
        return self.calibrator.predict_proba(model_input)


def log_calibrated_model(
    X: np.ndarray | pd.DataFrame,
    model: VennAbersCalibrator,
    threshold: float,
    model_registry_options: dict,
) -> ThresholdClassifier:
    """
    Wrap a calibrated model in a ThresholdClassifier and log it to MLflow.

    The model is wrapped with the tuned decision threshold, validated against
    a small sample of input data to confirm predictions are producible, and
    then logged to MLflow as a pyfunc model with an inferred signature.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Input features used to infer the MLflow model signature and generate
        sample predictions. Only the first 5 rows are used.

    model : VennAbersCalibrator
        A fitted Venn-Abers calibrated classifier. Must be already fitted;
        a NotFittedError will be raised otherwise.

    threshold : float
        Decision threshold for converting probabilities into class predictions.
        Must be in the range [0.0, 1.0].

    model_registry_options : dict
        Model registry configuration, see ModelRegistryConfig for fields.

    Returns
    -------
    ThresholdClassifier
        A ThresholdClassifier wrapping the calibrated model with the specified
        decision threshold, as logged to MLflow.

    Raises
    ------
    ValueError
        If threshold is not in the range [0.0, 1.0].
    NotFittedError
        If the provided model is not fitted.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be between 0 and 1")

    _ensure_fitted(model)

    threshold_classifier = ThresholdClassifier(model, threshold)
    sample_input = X[:5]

    with np.errstate(all="ignore"):
        sample_predictions = threshold_classifier.predict(sample_input)

    model_signature = infer_signature(
        model_input=X,
        model_output=sample_predictions
    )

    model_log_cfg = ModelRegistryConfig.from_params(model_registry_options)
    model_name = model_log_cfg.model_name

    logged_model_info = mlflow.pyfunc.log_model(
        name=model_name,
        python_model=threshold_classifier,
        signature=model_signature,
        input_example=sample_input,
        model_type="classifier",
    )

    logger.info("Logged model ID: %s", logged_model_info.model_id)
    logger.info("Logged model run ID: %s", logged_model_info.run_id)

    return threshold_classifier, logged_model_info

