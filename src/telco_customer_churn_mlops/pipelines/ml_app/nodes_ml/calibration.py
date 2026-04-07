import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mapie.calibration import VennAbersCalibrator
from matplotlib.figure import Figure
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibrationDisplay
from sklearn.utils.validation import check_X_y

from .utils import _ensure_fitted


def calibrate_fitted_classifer(
    model: ClassifierMixin,
    X_calib: np.ndarray | pd.DataFrame,
    y_calib: np.ndarray | pd.Series,
    calib_options: dict
) -> VennAbersCalibrator:
    """
    Calibrate a pre-fitted classifier using Venn-Abers calibration.

    This function assumes that the input `model` is already fitted.
    It applies Venn-Abers calibration on the provided calibration dataset.

    Parameters
    ----------
    model : ClassifierMixin
        A pre-fitted scikit-learn compatible classifier.

    X_calib : np.ndarray or pd.DataFrame
        Feature matrix used for calibration.

    y_calib : np.ndarray or pd.Series
        Target labels corresponding to `X_calib`.

    calib_options : Mapping[str, Any]
        Dictionary of calibration options:
            - cv : str or int, default="prefit"
                Cross-validation strategy. Use "prefit" when the model is already trained.
            - inductive : bool, default=False
                Whether to use inductive Venn-Abers calibration.
            - random_state : int, default=1071
                Random seed for reproducibility.

    Returns
    -------
    VennAbersCalibrator
        A fitted Venn-Abers calibrator wrapping the input model.

    Raises
    ------
    ValueError
        If the input model is not fitted.
    """
    cv = calib_options.get("cv", "prefit")
    inductive = calib_options.get("inductive", False)
    random_state = calib_options.get("random_state", 1071)

    # try:
    #     check_is_fitted(model)
    # except Exception as e:
    #     raise ValueError(
    #         "The model appears to be unfitted. Call `fit()` before using this function."
    #     ) from e
    _ensure_fitted(model)

    X_validated, y_validated = check_X_y(
        X=X_calib,
        y=y_calib,
        ensure_all_finite='allow-nan',
        accept_sparse=True
    )

    va_calibrator = VennAbersCalibrator(
        estimator=model,
        cv=cv,
        inductive=inductive,
        random_state=random_state
    )
    va_calibrator.fit(X_validated, y_validated)

    return va_calibrator



def plot_calibration_comparison(
    uncalibrated_model: ClassifierMixin,
    calibrated_model: VennAbersCalibrator,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
) -> Figure:
    """
    Plot calibration curves for uncalibrated and calibrated models.

    This function generates side-by-side calibration plots comparing the
    probability estimates of an uncalibrated classifier and a calibrated
    model (e.g., Venn-Abers). It uses predicted probabilities from both
    models and visualizes their calibration performance.

    Parameters
    ----------
    uncalibrated_model : ClassifierMixin
        A fitted scikit-learn compatible classifier that supports
        ``predict_proba``.

    calibrated_model : VennAbersCalibrator
        A fitted calibrated model that supports ``predict_proba``.

    X : np.ndarray or pd.DataFrame
        Feature matrix used for generating predictions.

    y : np.ndarray or pd.Series
        Ground truth target labels.

    n_bins : int, default=10
        Number of bins to discretize the [0, 1] interval for calibration
        curve estimation.

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib figure containing the calibration plots.

    Raises
    ------
    ValueError
        If either model does not support ``predict_proba``.

    Notes
    -----
    - Both models are assumed to be already fitted.
    - This function is suitable for use inside a Kedro pipeline node.
      The returned figure can be saved using a MatplotlibDataset.
    """

    _ensure_fitted(uncalibrated_model)
    _ensure_fitted(calibrated_model)

    if not hasattr(uncalibrated_model, "predict_proba"):
        raise ValueError("Uncalibrated model must support `predict_proba`.")

    if not hasattr(calibrated_model, "predict_proba"):
        raise ValueError("Calibrated model must support `predict_proba`.")

    uncalib_prob = uncalibrated_model.predict_proba(X)[:, 1]
    calib_prob = calibrated_model.predict_proba(X)[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    CalibrationDisplay.from_predictions(
        y,
        uncalib_prob,
        name="Uncalibrated",
        ax=axes[0],
        n_bins=n_bins,
    )

    CalibrationDisplay.from_predictions(
        y,
        calib_prob,
        name="Calibrated",
        ax=axes[1],
        n_bins=n_bins,
    )

    plt.tight_layout()
    plt.close(fig)
    return fig
