"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://docs.kedro.org/en/stable/configure/configuration_basics/#configuration"""

from telco_customer_churn_mlops.hooks import InferencePipelineRegistrationHook

# import mlflow
# from kedro.framework.hooks import hook_impl
# from kedro_mlflow.pipeline.pipeline_ml import PipelineML

# class DebugHook:
#     @hook_impl
#     def after_pipeline_run(
#         self,
#         run_params: dict,
#         pipeline,
#         catalog,
#         **kwargs,
#     ) -> None:
#         print("=" * 60)
#         print("DEBUG HOOK FIRED — after_pipeline_run")
#         print(f"pipeline_name : {run_params.get('pipeline_name')}")
#         print(f"pipeline type : {type(pipeline)}")

#         print(pipeline.outputs())
#         print(f"Is PipelineML : {isinstance(pipeline, PipelineML)}")

#         # check if challenger_beats_champion is accessible in catalog
#         try:
#             result = catalog.load("challenger_beats_champion")
#             print(f"challenger_beats_champion : {result}")
#         except Exception as e:
#             print(f"challenger_beats_champion not in catalog : {e}")

#         # check active mlflow run
        
#         active_run = mlflow.active_run()
#         run_id = active_run.info.run_id
#         model_uri = f"runs:/{run_id}/churn_inference_pipeline"
#         inf_pipeline = mlflow.pyfunc.load_model(model_uri=model_uri)
#         print(inf_pipeline)
#         print(f"active MLflow run : {active_run.info.run_id if active_run else 'None'}")
#         print("=" * 60)

# Instantiated project hooks.
# For example, after creating a hooks.py and defining a ProjectHooks class there, do
# from telco_customer_churn_mlops.hooks import ProjectHooks
# Hooks are executed in a Last-In-First-Out (LIFO) order.
HOOKS = (InferencePipelineRegistrationHook(),)

# Installed plugins for which to disable hook auto-registration.
# DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)

# Class that manages storing KedroSession data.
# from kedro.framework.session.store import BaseSessionStore
# SESSION_STORE_CLASS = BaseSessionStore
# Keyword arguments to pass to the `SESSION_STORE_CLASS` constructor.
# SESSION_STORE_ARGS = {
#     "path": "./sessions"
# }

# Directory that holds configuration.
# CONF_SOURCE = "conf"

# Class that manages how configuration is loaded.
# from kedro.config import OmegaConfigLoader

# CONFIG_LOADER_CLASS = OmegaConfigLoader

# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    # "config_patterns": {
    #     "spark" : ["spark*/"],
    #     "parameters": ["parameters*", "parameters*/**", "**/parameters*"],
    # }
}

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog
