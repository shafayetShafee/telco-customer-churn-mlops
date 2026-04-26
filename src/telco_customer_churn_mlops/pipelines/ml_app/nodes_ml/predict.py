import logging

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities.model_registry import ModelVersion

from .log_model import ThresholdClassifier

logger = logging.getLogger(__name__)


def infer_from_model(
    registered_model_version: ModelVersion,
    processed_infer_data: pd.DataFrame | np.ndarray,
    inference_options: dict,
) -> pd.DataFrame:
    """
    Load the registered champion model and run inference on processed data.

    The champion model is a ThresholdClassifier wrapping a Venn-Abers
    calibrated XGBoost model. Binary predictions are produced via the
    model's tuned decision threshold, and calibrated probabilities are
    obtained via the underlying ThresholdClassifier's predict_proba.

    Parameters
    ----------
    registered_model_version: mlflow.entities.model_registry.ModelVersion
        The champion ModelVersion object — either the newly registered
        version if the challenger was promoted, or the existing champion
        if registration was skipped.

    processed_infer_data : pd.DataFrame or np.ndarray
        Preprocessed feature matrix ready for inference. Must match the
        schema the model was trained on. If passed as np.ndarray, it is
        converted to a DataFrame internally before inference.

    inference_options : dict
        Inference configuration:
            - prediction_col (str): name of the prediction output column,
              default "churn_prediction"
            - proba_col (str): name of the probability output column,
              default "churn_probability"

    Returns
    -------
    pd.DataFrame
        Input data with two additional columns:
            - ``churn_prediction`` (int): binary class prediction (0 or 1)
              produced by the ThresholdClassifier.
            - ``churn_probability`` (float): calibrated positive-class
              probability from the underlying Venn-Abers calibrator.
    """
    target_col = inference_options.get("prediction_col", "churn_prediction")
    proba_col = inference_options.get("proba_col", "churn_probability")

    if isinstance(processed_infer_data, np.ndarray):
        infer_df = pd.DataFrame(processed_infer_data)
    else:
        infer_df = processed_infer_data

    logger.info(
        "Using registered model — name: %s | version: %s | model_id: %s \n"
        "(generated from run_id: %s )",
        registered_model_version.name,
        registered_model_version.version,
        registered_model_version.model_id,
        registered_model_version.run_id
    )

    champion_model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{registered_model_version.name}/{registered_model_version.version}",
        suppress_warnings=True,
    )
    threshold_classifier: ThresholdClassifier = champion_model.unwrap_python_model()

    predictions  = champion_model.predict(infer_df)
    probabilities = threshold_classifier.predict_proba(infer_df)[:, 1]

    return infer_df.assign(
        **{
            target_col: predictions,
            proba_col: probabilities,
        }
    )
