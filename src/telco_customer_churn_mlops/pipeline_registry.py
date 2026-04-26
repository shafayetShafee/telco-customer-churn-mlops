"""Project pipelines."""

from __future__ import annotations

import logging
import os

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
from kedro.pipeline import Pipeline
from kedro_mlflow.pipeline import pipeline_ml_factory
from rich.pretty import pretty_repr

from telco_customer_churn_mlops.configs import ModelRegistryConfig
from telco_customer_churn_mlops.pipelines.etl_app.pipeline import create_etl_pipeline
from telco_customer_churn_mlops.pipelines.ml_app.pipeline import create_ml_pipeline
from telco_customer_churn_mlops.pipelines.user_app.pipeline import create_user_pipeline

logger = logging.getLogger(__name__)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    env = os.environ.get("KEDRO_ENV", "local")
    conf_loader = OmegaConfigLoader(conf_source=settings.CONF_SOURCE, env=env)
    model_reg_params = conf_loader["parameters"].get("model_registry_options", {})
    model_reg_config = ModelRegistryConfig.from_params(model_reg_params)

    if not model_reg_params:
        logger.warning(
            "No [bold dark_orange3]model_registry_options[/] found in [bold blue]parameters*.yml[/] "
            "in [bold magenta]conf/[/] directory. "
            "[bold yellow]Falling back to defaults[/]: \n%s",
            pretty_repr(model_reg_config.model_dump()),
            extra={"markup": True},
        )

    etl_pipeline = create_etl_pipeline()
    ml_pipeline = create_ml_pipeline()
    user_pipeline = create_user_pipeline()

    train_etl_pipeline = etl_pipeline.only_nodes_with_tags("training")
    training_pipeline = (etl_pipeline + ml_pipeline).only_nodes_with_tags("training")
    inference_pipeline = (etl_pipeline + ml_pipeline).only_nodes_with_tags("inference")

    training_pipeline_ml = pipeline_ml_factory(
        training=training_pipeline,
        inference=inference_pipeline,
        input_name="future_infer_data",
        log_model_kwargs=dict(
            name=model_reg_config.inference_pipeline_name,
            signature=None,
            registered_model_name=None,
        ),
    )

    pipelines = {
        "etl": train_etl_pipeline,
        "train": training_pipeline_ml,
        "user": user_pipeline,
        "__default__": train_etl_pipeline + training_pipeline,
    }
    return pipelines
