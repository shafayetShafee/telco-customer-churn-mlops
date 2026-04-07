import logging

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import classification_report
from sklearn.utils.validation import check_X_y
from mapie.calibration import VennAbersCalibrator

from xgboost import XGBClassifier

from .utils import _ensure_fitted

logger = logging.getLogger(__name__)


def fit_calibrated_final_model(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    best_params: dict,
    calib_options: dict
) -> VennAbersCalibrator:
    """
    Fit an XGBoost classifier on the full dataset and calibrate using Venn-Abers (MAPIE).

    This function initializes an XGBClassifier with the provided hyperparameters,
    validates the input data, and fits a Venn-Abers calibrated classifier.

    Parameters
    ----------
    X : Union[pd.DataFrame, np.ndarray]
        Feature matrix for training.
    y : Union[pd.Series, np.ndarray]
        Target vector for training.
    best_params : Dict
        Optimized hyperparameters for XGBoost (e.g., from Optuna or GridSearch).
    calib_options : Dict
        Calibration options for Venn-Abers:
            - 'inductive' (bool): whether to use inductive Venn-Abers (default False)
            - 'n_splits' (int): number of CV folds for calibration (default 5)
            - 'random_state' (int): random seed for reproducibility (default 1071)

    Returns
    -------
    VennAbersCalibrator
        A fitted Venn-Abers calibrated classifier.

    Raises
    ------
    ValueError
        If X or y are None or empty.
    """
    if X is None or y is None:
        raise ValueError("X and y must not be None")
    
    n_splits = calib_options.get("n_splits", 5)
    random_state = calib_options.get("random_state", 1071)

    X_validated, y_validated = check_X_y(
        X=X,
        y=y,
        ensure_all_finite='allow-nan',
        accept_sparse=True
    )

    va_calibrator = VennAbersCalibrator(
        estimator=XGBClassifier(**best_params),
        cv=None,
        inductive=False,
        n_splits=n_splits,
        random_state=random_state
    )

    va_calibrator.fit(X_validated, y_validated)
    return va_calibrator



def evaluate_model(
    model: ClassifierMixin | VennAbersCalibrator,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    threshold: float
) -> pd.DataFrame:
    """
    Evaluate a fitted classification model using a custom decision threshold.

    This function validates the model, generates probability predictions, 
    applies a custom threshold, computes a classification report, logs it, 
    and returns it as a DataFrame.

    Parameters
    ----------
    model : ClassifierMixin | VennAbersCalibrator
        A fitted classification or calibrated model supporting `predict_proba`.
    
    X_test : pd.DataFrame or np.ndarray
        Test features.
    
    y_test : pd.Series or np.ndarray
        True labels for the test data.
    
    threshold : float
        Decision threshold for converting probabilities into class predictions.
        Must be between 0 and 1.

    Returns
    -------
    pd.DataFrame
        Structured classification report with columns:
        - class: label
        - precision
        - recall
        - f1-score
        - support
        - accuracy / macro avg / weighted avg (from scikit-learn)
    
    Raises
    ------
    ValueError
        If `threshold` is not between 0 and 1, or if `X_test` or `y_test` is None, 
        or if the model is not fitted.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be between 0 and 1")

    if X_test is None or y_test is None:
        raise ValueError("X_test and y_test must not be None")

    _ensure_fitted(model)

    X_valid, y_valid = check_X_y(
        X_test,
        y_test,
        ensure_all_finite='allow-nan',
        accept_sparse=True
    )
    y_proba = model.predict_proba(X_valid)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    report_str = classification_report(y_valid, y_pred)
    report_dict = classification_report(y_valid, y_pred, output_dict=True)

    logger.info("Classification Report:\n%s", report_str)

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.index.name = "class"
    report_df = report_df.reset_index()
    return report_df
