import logging

import mlflow
import pandas as pd
import shap

from telco_customer_churn_mlops.configs import (
    InferenceConfig,
    ModelRegistryConfig,
    ShapConfig,
)
from telco_customer_churn_mlops.pipelines.ml_app.nodes_ml.log_shap_explanation import (
    _build_shap_plots,
)
from telco_customer_churn_mlops.pipelines.ml_app.nodes_ml.model_registry import (
    _get_champion_version,
)

logger = logging.getLogger(__name__)


def serve_predictions(
    raw_new_data: pd.DataFrame,
    model_registry_options: dict,
) -> pd.DataFrame:
    """
    Load the latest registered version of the churn inference pipeline
    from MLflow and run inference on raw future data.

    Parameters
    ----------
    raw_new_data : pd.DataFrame
        Raw future data to run inference on. Must match the schema of the
        training data before preprocessing — the inference pipeline handles
        all preprocessing internally.

    model_registry_options : dict
        Model registry configurations. Used to load the inference pipeline from
        Mlflow model registry. See ModelRegistryConfig for fields.

    Returns
    -------
    pd.DataFrame
        Prediction results containing binary churn predictions and
        calibrated churn probabilities.
    """
    registry_cfg = ModelRegistryConfig.from_params(model_registry_options)
    inference_pipeline_name = registry_cfg.inference_pipeline_name
    inference_pipeline_alias = registry_cfg.champion_alias

    champion_version = _get_champion_version(
        model_name=inference_pipeline_name, champion_alias=inference_pipeline_alias
    )

    if champion_version is None:
        raise RuntimeError(
            f"No champion version found for '{inference_pipeline_name}'. "
            "Ensure the training pipeline has been run at least once."
        )

    logger.info(
        "Loaded the %s pipeline — version: %s | model_uri: %s\nrun_id: %s",
        champion_version.name,
        champion_version.version,
        champion_version.source,
        champion_version.run_id,
    )

    model_uri = f"models:/{inference_pipeline_name}@{inference_pipeline_alias}"
    inference_pipeline = mlflow.pyfunc.load_model(
        model_uri=model_uri,
        suppress_warnings=True,
    )

    return inference_pipeline.predict(raw_new_data)


def serve_shap_values(
    prediction_results: pd.DataFrame,
    inference_options: dict,
    explainer_registry_options: dict,
    shap_options: dict,
) -> shap.Explanation:
    """
    Load the champion SHAP explainer from the MLflow model registry and
    compute SHAP values for the prediction results.

    Parameters
    ----------
    prediction_results : pd.DataFrame
        Output from serve_predictions — contains feature columns alongside
        prediction and probability columns, which are dropped before
        computing SHAP values.

    inference_options : dict
        Inference configuration. Used to identify and drop prediction and
        probability columns. See InferenceConfig for fields.

    explainer_registry_options : dict
        Registry configuration for the SHAP explainer. Used to load the
        champion explainer from MLflow model registry.
        See ModelRegistryConfig for fields.

    shap_options : dict
        SHAP configurations carrying all SHAP-related hyperparameters.
        See ShapConfig for fields.

    Returns
    -------
    shap.Explanation
        SHAP explanation object containing shap values, base values, and
        feature data for each prediction in the input DataFrame.

    Raises
    ------
    RuntimeError
        If no champion version is found for the SHAP explainer — ensure
        the training pipeline has been run at least once.
    """
    infer_cfg = InferenceConfig.from_params(inference_options)
    exp_reg_cfg = ModelRegistryConfig.from_params(explainer_registry_options)
    shap_config = ShapConfig.from_params(shap_options)

    explainer_name = exp_reg_cfg.model_name
    explainer_alias = exp_reg_cfg.champion_alias

    infer_df = prediction_results.drop(
        columns=[infer_cfg.prediction_col, infer_cfg.proba_col]
    )

    champion_version = _get_champion_version(
        model_name=explainer_name, champion_alias=explainer_alias
    )

    if champion_version is None:
        raise RuntimeError(
            f"No champion version found for '{explainer_name}'. "
            "Ensure the training pipeline has been run at least once."
        )

    logger.info(
        "Loaded the '%s' pipeline — version: %s | model_uri: %s\nrun_id: %s",
        champion_version.name,
        champion_version.version,
        champion_version.source,
        champion_version.run_id,
    )

    model_uri = f"models:/{explainer_name}@{explainer_alias}"
    explainer = mlflow.shap.load_explainer(model_uri=model_uri)

    shap_values = explainer(infer_df)
    max_display = min(infer_df.shape[1], shap_config.plots_max_display)
    bar_fig, beeswarm_fig = _build_shap_plots(shap_values, max_display)

    return shap_values, bar_fig, beeswarm_fig
