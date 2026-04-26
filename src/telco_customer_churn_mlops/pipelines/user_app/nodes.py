import logging

import mlflow
import pandas as pd

logger = logging.getLogger(__name__)


def serve_predictions(
    raw_new_data: pd.DataFrame
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

    Returns
    -------
    pd.DataFrame
        Prediction results containing binary churn predictions and
        calibrated churn probabilities.
    """
    client = mlflow.MlflowClient()
    versions = client.search_model_versions("name='churn_inference_pipeline'")

    if not versions:
        raise RuntimeError(
            "No versions found for registered model 'churn_inference_pipeline'. "
            "Ensure the training pipeline has been run at least once."
        )

    latest_version = max(versions, key=lambda mv: int(mv.version))

    logger.info(
        "Loading churn inference pipeline — version: %s | model_uri: %s\n"
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

