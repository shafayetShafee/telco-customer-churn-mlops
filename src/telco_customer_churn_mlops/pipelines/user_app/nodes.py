import logging

import mlflow
import pandas as pd

from telco_customer_churn_mlops.configs import ModelRegistryConfig

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

    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{inference_pipeline_name}'")

    if not versions:
        raise RuntimeError(
            f"No versions found for registered model '{inference_pipeline_name}'. "
            "Ensure the training pipeline has been run at least once."
        )

    latest_version = max(versions, key=lambda mv: int(mv.version))

    logger.info(
        "Loading the inference pipeline — version: %s | model_uri: %s\n"
        "run_id: %s",
        latest_version.version,
        latest_version.source,
        latest_version.run_id,
    )

    inference_pipeline = mlflow.pyfunc.load_model(
        model_uri=f"models:/{latest_version.name}/{latest_version.version}",
        suppress_warnings=True,
    )

    return inference_pipeline.predict(raw_new_data)

