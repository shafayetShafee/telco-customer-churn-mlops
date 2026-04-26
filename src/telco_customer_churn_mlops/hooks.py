import logging
from typing import Any

import mlflow
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)

INFERENCE_PIPELINE_NAME = "churn_inference_pipeline"
MODEL_NAME = "calibrated_threshold_classifier"
CHAMPION_ALIAS = "champion"


class InferencePipelineRegistrationHook:
    """
    Kedro hook that conditionally registers the logged churn inference
    pipeline to the MLflow Model Registry after the training pipeline
    completes.

    Registration only occurs if the challenger beat the champion during
    evaluation (or always_replace=True). This keeps inference pipeline
    versions in sync with promoted champion model versions, avoiding
    redundant registry versions from unsuccessful training runs.

    The hook reads ``challenger_beats_champion`` from the Kedro data
    catalog after pipeline execution.
    """

    @hook_impl
    def after_pipeline_run(
        self,
        run_params: dict[str, Any],
        pipeline,
        catalog: DataCatalog,
    ) -> None:
        """
        Conditionally trigger inference pipeline registration after the
        training pipeline completes.

        Skips registration silently if:
            - The pipeline is not the training pipeline
            - ``challenger_beats_champion`` is not in the catalog
            - The challenger did not beat the champion

        Parameters
        ----------
        run_params : dict[str, Any]
            Kedro run parameters including the pipeline name.
        pipeline : Pipeline
            The Kedro pipeline that was executed.
        catalog : DataCatalog
            The Kedro data catalog containing all node outputs, including
            ``challenger_beats_champion``.
        """
        pipeline_name = run_params.get("pipeline_name", "__default__")
        if pipeline_name not in ("train", "__default__"):
            logger.info(
                "Skipping inference pipeline registration for pipeline: %s",
                pipeline_name,
            )
            return None

        try:
            challenger_beats_champion = catalog.load("challenger_beats_champion")
        except Exception:
            logger.info(
                "challenger_beats_champion not found in catalog — "
                "skipping inference pipeline registration."
            )
            return None

        if not challenger_beats_champion:
            logger.info(
                "Challenger did not beat champion — "
                "skipping inference pipeline registration."
            )
            return None

        self._register_inference_pipeline()

    def _register_inference_pipeline(self) -> None:
        """
        Register the logged churn inference pipeline artifact to the MLflow
        Model Registry and tag it with the current champion model version.

        Skips registration if:
            - No active MLflow run is found
            - No champion model is registered under the champion alias
            - MLflow registration fails
        """
        client = MlflowClient()

        active_run = mlflow.active_run()
        if active_run is None:
            logger.warning(
                "No active MLflow run found — cannot register inference pipeline."
            )
            return None

        try:
            champion_model_version = client.get_model_version_by_alias(
                MODEL_NAME, CHAMPION_ALIAS
            )
        except MlflowException:
            logger.warning(
                "No champion model registered under alias '%s' — "
                "skipping inference pipeline registration.",
                CHAMPION_ALIAS,
            )
            return None

        run_id = active_run.info.run_id
        model_uri = f"runs:/{run_id}/{INFERENCE_PIPELINE_NAME}"

        logger.info(
            "Registering churn inference pipeline from run %s", run_id
        )

        try:
            mv = mlflow.register_model(
                model_uri=model_uri,
                name=INFERENCE_PIPELINE_NAME,
            )
        except MlflowException as e:
            logger.error("Failed to register inference pipeline: %s", e)
            return None

        client.set_model_version_tag(
            INFERENCE_PIPELINE_NAME,
            mv.version,
            "champion_model_version",
            champion_model_version.version,
        )
        client.set_registered_model_alias(
            INFERENCE_PIPELINE_NAME, CHAMPION_ALIAS, mv.version
        )

        logger.info(
            "Registered churn inference pipeline — version: %s | run_id: %s\n"
            "champion_model_version: %s",
            mv.version,
            run_id,
            champion_model_version.version,
        )