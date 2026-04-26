"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline
from kedro_mlflow.pipeline import pipeline_ml_factory

from telco_customer_churn_mlops.pipelines.etl_app.pipeline import create_etl_pipeline
from telco_customer_churn_mlops.pipelines.ml_app.pipeline import create_ml_pipeline
from telco_customer_churn_mlops.pipelines.user_app.pipeline import create_user_pipeline


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
        input_name='future_infer_data',
        log_model_kwargs=dict(
            name="churn_inference_pipeline",
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

