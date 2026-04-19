"""Project pipelines."""
from __future__ import annotations
from platform import python_version

from kedro.pipeline import Pipeline
from kedro_mlflow.pipeline import pipeline_ml_factory

from telco_customer_churn_mlops import __version__ as PROJECT_VERSION
from telco_customer_churn_mlops.pipelines.etl_app.pipeline import create_etl_pipeline
from telco_customer_churn_mlops.pipelines.ml_app.pipeline import create_ml_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
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
        input_name='telco',
        log_model_kwargs=dict(
            artifact_path="telco_customer_churn_mlops"
        )
    )

    pipelines = {
        "etl_app": etl_pipeline,
        "training_pipeline": training_pipeline_ml,
        "inference_pipeline": inference_pipeline,
        "__default__": etl_pipeline 
        + inference_pipeline
    }
    return pipelines
