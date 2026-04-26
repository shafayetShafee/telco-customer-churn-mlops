"""Project pipelines."""
from __future__ import annotations

import os

from kedro.pipeline import Pipeline
from kedro_mlflow.pipeline import pipeline_ml_factory
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

from telco_customer_churn_mlops.pipelines.etl_app.pipeline import create_etl_pipeline
from telco_customer_churn_mlops.pipelines.ml_app.pipeline import create_ml_pipeline
from telco_customer_churn_mlops.pipelines.user_app.pipeline import create_user_pipeline
from telco_customer_churn_mlops.configs import ModelRegistryConfig


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    env = os.environ.get("KEDRO_ENV", "local")
    conf_loader = OmegaConfigLoader(conf_source=settings.CONF_SOURCE, env=env)
    model_reg_params = conf_loader["parameters"].get("model_registry_options")

    if model_reg_params is None:
        raise KeyError(
            "Missing 'model_registry_options' in parameters*.yml in `conf/` directory. "
            "Please define `model_name`, `champion_alias`, and `inference_pipeline_name`."
        )
    
    model_reg_config = ModelRegistryConfig.from_params(model_reg_params)

    etl_pipeline = create_etl_pipeline()
    ml_pipeline = create_ml_pipeline()

    training_pipeline = (
        etl_pipeline
        + ml_pipeline
    ).only_nodes_with_tags('training')

    inference_pipeline = (
        etl_pipeline
        + ml_pipeline
    ).only_nodes_with_tags('inference')

    training_pipeline_ml = pipeline_ml_factory(
        training=training_pipeline,
        inference=inference_pipeline,
        input_name='future_infer_data',
        log_model_kwargs=dict(
            name=model_reg_config.inference_pipeline_name,
            signature=None,
            registered_model_name=None
        )
    )

    train_etl_pipeline = etl_pipeline.only_nodes_with_tags('training')
    user_pipeline = create_user_pipeline()

    pipelines = {
        "etl": train_etl_pipeline,
        "train": training_pipeline_ml,
        "user": user_pipeline,
        "__default__": train_etl_pipeline + training_pipeline
    }
    return pipelines

