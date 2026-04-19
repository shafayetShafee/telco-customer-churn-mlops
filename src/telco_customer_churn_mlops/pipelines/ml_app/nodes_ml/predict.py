import pandas as pd
import numpy as np

from sklearn.base import ClassifierMixin
from mapie.calibration import VennAbersCalibrator

from .utils import _ensure_fitted


def infer_from_model(
    model: ClassifierMixin | VennAbersCalibrator,
    X: pd.DataFrame | np.ndarray
) -> pd.Series:
    """
    Generate predicted probabilities from a fitted calibrated model.

    Parameters
    ----------
    model : ClassifierMixin | VennAbersCalibrator
        A fitted classification or calibrated model supporting `predict_proba`.
    
    X : pd.DataFrame or np.ndarray
        Preprocessed feature matrix or dataframe.

    Returns
    -------
    pd.Series
        Predicted probabilities for the positive class (churn).
    """

    if model is None:
        raise ValueError("Model must not be None")

    if X is None or len(X) == 0:
        raise ValueError("Input features must not be None or empty")
    
    _ensure_fitted(model)

    proba = model.predict_proba(X)
    churn_proba = proba[:, 1]

    return pd.Series(churn_proba, name="churn_probability")


def decode_predictions(
    probabilities: pd.Series,
    threshold: float
) -> pd.Series:
    """
    Convert predicted probabilities into class labels using a threshold.

    Parameters
    ----------
    probabilities : pd.Series
        Predicted probabilities for the positive class.
        
    threshold : float
        Decision threshold.

    Returns
    -------
    pd.Series
        Predicted labels: "churn" or "not-churn".
    """

    if probabilities is None or len(probabilities) == 0:
        raise ValueError("Probabilities must not be None or empty")

    if not (0 <= threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1")

    predictions = probabilities.apply(
        lambda p: "churn" if p > threshold else "not-churn"
    )

    return predictions.rename("churn_prediction")