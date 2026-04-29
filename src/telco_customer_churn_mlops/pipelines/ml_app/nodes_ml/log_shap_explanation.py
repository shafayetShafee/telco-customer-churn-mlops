import logging
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
import shap

from mlflow.models.model import ModelInfo
from telco_customer_churn_mlops.configs import ShapConfig

logger = logging.getLogger(__name__)


def log_shap_explanations(
    logged_model_info: ModelInfo,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    config: dict
) -> tuple[ModelInfo, plt.Figure, plt.Figure]:
    """
    Build a SHAP explainer for a logged MLflow PythonModel, compute SHAP values
    on an evaluation set, log the explainer to MLflow, and return summary plots.

    The explainer is built against the positive-class predicted probability
    (``model.predict_proba(X)[:, 1]``), so all SHAP values are in probability
    space and directly interpretable as probability contributions.

    Background data is sampled from ``X_train`` (the distribution the model was
    trained on), which is the statistically correct reference for the masker.

    Parameters
    ----------
    model_info : ModelInfo
        MLflow ``ModelInfo`` object returned when the model was logged.

    X_train : pd.DataFrame
        Training features. Sampled to construct the SHAP background masker.

    X_test : pd.DataFrame
        Held-out features. Used as the evaluation set for SHAP value computation.
        Optionally down-sampled via ``config.eval_data_size``.

    config : ShapConfig
        Dataclass carrying all SHAP-related hyperparameters:
        ``bg_data_size``, ``eval_data_size``, ``random_state``,
        ``explainer_name``, and ``plots_max_display``.

    Returns
    -------
    explainer_info : ModelInfo
        MLflow ``ModelInfo`` for the logged SHAP explainer artifact.

    bar_fig : plt.Figure
        Mean absolute SHAP value bar chart (global feature importance).

    beeswarm_fig : plt.Figure
        SHAP beeswarm plot (feature importance + direction + distribution).

    Raises
    ------
    ValueError
        If ``X_train`` or ``X_test`` is empty.
    mlflow.exceptions.MlflowException
        If the model cannot be loaded from ``model_info.model_uri``
    """
    if X_train.empty or X_test.empty:
        raise ValueError("X_train and X_test must not be empty.")
    
    logger.info("Building SHAP Explainer for model ID: %s", logged_model_info.model_id)
    
    mlflow_model = mlflow.pyfunc.load_model(
        model_uri=logged_model_info.model_uri, 
        suppress_warnings=True,
    )
    model = mlflow_model.unwrap_python_model()

    shap_config = ShapConfig.from_params(config)
    logger.info("Building SHAP explainer (background size: %d)", shap_config.bg_data_size)

    bg_data = shap.utils.sample(
        X=X_train,
        nsamples=shap_config.bg_data_size,
        random_state=shap_config.random_state
    )

    predict_fn_pos = lambda x: model.predict_proba(x)[:, 1]

    explainer = shap.Explainer(
        model=predict_fn_pos, 
        masker=bg_data, 
        seed=shap_config.random_state
    )

    explainer_info = mlflow.shap.log_explainer(
        explainer=explainer,
        name=shap_config.explainer_name,
        serialize_model_using_mlflow=True,
        signature=None
    )

    logger.info("Logged explainer ID: %s", explainer_info.model_id)
    logger.info("Logged explainer run ID: %s", explainer_info.run_id)

    eval_data = (
        shap.utils.sample(X_test, shap_config.eval_data_size, random_state=shap_config.random_state)
        if shap_config.eval_data_size
        else X_test
    )

    logger.info("Computing SHAP values on %d samples", len(eval_data))
    shap_values = explainer(eval_data)

    max_display = min(X_test.shape[1], shap_config.plots_max_display)

    bar_fig, beeswarm_fig = _build_shap_plots(shap_values, max_display)

    return explainer_info, bar_fig, beeswarm_fig


def _build_shap_plots(
    shap_values: shap.Explanation,
    max_display: int,
) -> tuple[plt.Figure, plt.Figure]:
    """
    Render SHAP summary plots as standalone Matplotlib figures.

    Parameters
    ----------
    shap_values : shap.Explanation
        Explanation object returned by ``explainer(eval_data)``.

    max_display : int
        Maximum number of features to display in each plot.

    Returns
    -------
    bar_fig : plt.Figure
        Global feature importance bar chart.
    beeswarm_fig : plt.Figure
        Beeswarm summary plot.
    """
    bar_fig, ax = plt.subplots(tight_layout=True, figsize=(12, 6))
    shap.plots.bar(shap_values, max_display=max_display, show=False, ax=ax)
    ax.set_title("SHAP Feature Importance (Mean |SHAP|)")

    beeswarm_fig, ax = plt.subplots(tight_layout=True, figsize=(12, 6))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False, ax=ax, plot_size = None)
    ax.set_title("SHAP Beeswarm — Feature Impact Distribution")

    return bar_fig, beeswarm_fig
