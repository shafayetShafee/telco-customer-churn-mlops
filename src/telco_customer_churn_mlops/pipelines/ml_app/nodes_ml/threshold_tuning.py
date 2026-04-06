from typing import (
    Iterable, Optional, Union, Tuple
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.utils.validation import check_is_fitted


def _evaluate_thresholds(
    model: ClassifierMixin,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    thresholds: Optional[Iterable[float]] = None,
    pos_label: Union[int, str] = 1,
    use_proba: bool = True,
    zero_division: int = 0
) -> pd.DataFrame:
    """
    Evaluate classification performance across multiple decision thresholds.

    This function computes common classification metrics for a range of
    probability (or score) thresholds. It supports models that provide either
    `predict_proba` or `decision_function`, and evaluates how metrics such as
    precision, recall, F1-score, TPR, FPR, and accuracy vary with the threshold.

    Args:
        model (ClassifierMixin):
            A fitted scikit-learn compatible classification model.
        X (Union[np.ndarray, pd.DataFrame]):
            Feature data used for generating predictions.
        y (Union[np.ndarray, pd.Series]):
            Ground truth target labels.
        thresholds (Optional[Iterable[float]]):
            Iterable of threshold values to evaluate. If None, defaults to
            101 evenly spaced values between 0 and 1.
        pos_label (Union[int, str]):
            The label considered as the positive class.
        use_proba (bool):
            If True, use `predict_proba` to obtain scores. If False, use
            `decision_function` and normalize scores to [0, 1].
        zero_division (int):
            Value to return when there is a zero division in precision or recall.

    Returns:
        pd.DataFrame:
            DataFrame containing evaluation metrics for each threshold with columns:
            - threshold: decision threshold
            - precision: precision score
            - recall: recall score
            - f1: F1-score
            - tpr: true positive rate (recall)
            - fpr: false positive rate
            - accuracy: classification accuracy
    """
    try:
        check_is_fitted(model)
    except Exception as e:
        raise ValueError(
            "The model appears to be unfitted. Call `fit()` before using this function."
        ) from e

    if use_proba:
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)[:, 1]
        else:
            raise ValueError("Model does not support predict_proba")
    else:
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)

            denom = scores.max() - scores.min()
            scores = (scores - scores.min()) / denom if denom > 0 else np.zeros_like(scores)
        else:
            raise ValueError("Model does not support decision_function")

    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    results = []
    
    for t in thresholds:
        y_pred = (scores >= t).astype(int)

        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        results.append({
            "threshold": float(t),
            "precision": precision_score(y, y_pred, pos_label=pos_label, zero_division=zero_division),
            "recall": recall_score(y, y_pred, pos_label=pos_label, zero_division=zero_division),
            "f1": f1_score(y, y_pred, pos_label=pos_label, zero_division=zero_division),
            "tpr": tpr,
            "fpr": fpr,
            "accuracy": accuracy_score(y, y_pred),
        })

    return pd.DataFrame(results)



def _select_threshold_by_recall(
    df: pd.DataFrame,
    min_recall: float = 0.8
) -> float:
    """
    Select an optimal decision threshold based on a minimum recall constraint.

    This function filters a DataFrame of threshold evaluation metrics to retain
    only rows where recall is greater than or equal to the specified minimum.
    Among the valid thresholds, it selects the highest threshold value to favor
    a more conservative decision boundary.

    Args:
        df (pd.DataFrame):
            DataFrame containing evaluation metrics for different thresholds.
            Must include at least the columns:
            - 'threshold': threshold values
            - 'recall': recall scores corresponding to each threshold
        min_recall (float):
            Minimum recall value required for threshold selection.

    Returns:
        float:
            Selected threshold value that satisfies the recall constraint.

    Raises:
        ValueError:
            If no threshold satisfies the minimum recall requirement.
    """

    valid = df[df["recall"] >= min_recall]

    if valid.empty:
        raise ValueError("No threshold satisfies the recall constraint")

    best_row = valid.sort_values("threshold", ascending=False).iloc[0]

    return float(best_row["threshold"])



def tune_threshold(
    model: ClassifierMixin,
    X_valid: Union[pd.DataFrame, np.ndarray],
    y_valid: Union[pd.Series, np.ndarray],
    min_recall: float
) -> Tuple[pd.DataFrame, float]:
    """
    Evaluate model performance across thresholds and select an optimal threshold
    based on a minimum recall constraint.

    This function computes classification metrics over a range of thresholds
    using the validation dataset, then selects the highest threshold that
    achieves at least the specified minimum recall.

    Args:
        model (ClassifierMixin):
            A fitted scikit-learn compatible classification model.
        X_valid (Union[pd.DataFrame, np.ndarray]):
            Validation feature data.
        y_valid (Union[pd.Series, np.ndarray]):
            Validation target labels.
        min_recall (float):
            Minimum recall value required for threshold selection. Must be
            between 0 and 1.

    Returns:
        Tuple[pd.DataFrame, float]:
            - df: DataFrame containing evaluation metrics across thresholds.
            - best_threshold: Selected threshold satisfying the recall constraint.

    Raises:
        ValueError:
            If `min_recall` is not between 0 and 1.
    """

    if not (0.0 <= min_recall <= 1.0):
        raise ValueError("min_recall must be between 0 and 1")

    df = _evaluate_thresholds(model, X_valid, y_valid)

    if df.empty:
        raise ValueError("Threshold evaluation returned an empty DataFrame")

    best_threshold = _select_threshold_by_recall(df, min_recall)

    return df, best_threshold



def plot_threshold_metrics(
    df: pd.DataFrame,
    best_threshold: Optional[float] = None,
    title: Optional[str] = "Threshold Tuning Metrics"
) -> Figure:
    """
    Generate a publication-quality plot of classification metrics across thresholds.

    Args:
        df (pd.DataFrame):
            DataFrame containing threshold evaluation results. Must include
            a 'threshold' column and metric columns (e.g., 'precision',
            'recall', 'f1', 'accuracy').
        title (Optional[str], optional):
            Title of the plot. Defaults to "Threshold Tuning Metrics".

    Returns:
        Figure:
            A Matplotlib Figure object ready for saving via Kedro catalog.

    Raises:
        ValueError:
            If required columns are missing or DataFrame is empty.
    """

    if df.empty:
        raise ValueError("Input DataFrame is empty")

    if "threshold" not in df.columns:
        raise ValueError("DataFrame must contain a 'threshold' column")

    plot_df = df.set_index("threshold")

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.plot(ax=ax, linewidth=2)

    if best_threshold is not None:
        ax.axvline(
            x=best_threshold,
            linestyle="--",
            linewidth=2,
            label=f"Selected Threshold ({best_threshold:.2f})"
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(
        title="Metrics",
        loc="best",
        frameon=True
    )
    fig.tight_layout()
    plt.close(fig)
    return fig