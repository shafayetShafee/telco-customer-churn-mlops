import logging

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import classification_report
from sklearn.utils.validation import check_is_fitted, check_X_y
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def train_xgb_model(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    best_params: dict
) -> XGBClassifier:
    """
    Train an XGBoost classifier using the provided training data and best hyperparameters.

    This function initializes an XGBClassifier with the given hyperparameters,
    validates the input data, and fits the model on the training dataset.

    Args:
        X_train (Union[pd.DataFrame, np.ndarray]):
            Training feature matrix.
        y_train (Union[pd.Series, np.ndarray]):
            Training target labels.
        best_params (Dict):
            Dictionary of optimized hyperparameters (e.g., from Optuna or GridSearch).

    Returns:
        XGBClassifier:
            A fitted XGBoost classification model.

    Raises:
        ValueError:
            If input data is invalid or empty.
    """
    if X_train is None or y_train is None:
        raise ValueError("X_train and y_train must not be None")

    X_validated, y_validated = check_X_y(
        X=X_train,
        y=y_train,
        ensure_all_finite='allow-nan',
        accept_sparse=True
    )

    model = XGBClassifier(**best_params)
    model.fit(X_validated, y_validated)
    return model



def evaluate_model(
    model: ClassifierMixin,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    threshold: float
) -> pd.DataFrame:
    """
    Evaluate a fitted classification model using a custom decision threshold.

    This function:
    - Validates that the model is fitted
    - Generates probability predictions
    - Applies a custom threshold to obtain class predictions
    - Computes a classification report
    - Logs the report using a logger
    - Returns the report as a DataFrame (for Kedro saving)

    Args:
        model (ClassifierMixin):
            A fitted scikit-learn compatible classification model.
        X_test (Union[pd.DataFrame, np.ndarray]):
            Test feature data.
        y_test (Union[pd.Series, np.ndarray]):
            Ground truth labels.
        threshold (float):
            Decision threshold for converting probabilities into class labels.
            Must be between 0 and 1.

    Returns:
        pd.DataFrame:
            Classification report as a structured DataFrame.
    """

    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be between 0 and 1")

    if X_test is None or y_test is None:
        raise ValueError("X_test and y_test must not be None")

    try:
        check_is_fitted(model)
    except Exception as e:
        raise ValueError(
            "The model appears to be unfitted. Call `fit()` before using this function."
        ) from e

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
