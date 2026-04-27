from mapie.calibration import VennAbersCalibrator
from sklearn.base import ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


def _ensure_fitted(model: ClassifierMixin | VennAbersCalibrator) -> None:
    """
    Ensure that the given model is fitted.

    This function checks whether a model has been fitted. It supports:
    - Scikit-learn classifiers (using `check_is_fitted`)
    - Custom calibrated models such as `VennAbersCalibrator` (using
      the `is_fitted` attribute)

    Parameters
    ----------
    model : ClassifierMixin | VennAbersCalibrator
        A fitted classification model or calibrated model. Must support
        either the scikit-learn fit interface or have an `is_fitted` attribute.

    Raises
    ------
    ValueError
        If the model is not fitted. The error message is:
        "The model appears to be unfitted. Call `fit()` before using this function."
    """
    not_fitted_msg = (
        "The model appears to be unfitted. Call `fit()` before using this function."
    )

    if hasattr(model, "is_fitted"):
        if not getattr(model, "is_fitted"):
            raise ValueError(not_fitted_msg)
    else:
        try:
            check_is_fitted(model)
        except NotFittedError as e:
            raise ValueError(not_fitted_msg) from e
