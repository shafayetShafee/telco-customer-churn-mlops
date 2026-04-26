from pydantic import BaseModel, ValidationInfo, field_validator


class ModelRegistryConfig(BaseModel):
    model_name: str = "calibrated_threshold_classifier"
    champion_alias: str = "champion"
    inference_pipeline_name: str = "churn_inference_pipeline"
    always_replace: bool = False
    eval_metric: str = "recall_score"

    @field_validator(
        "model_name",
        "champion_alias",
        "inference_pipeline_name",
        "eval_metric",
        mode="before",
    )
    @classmethod
    def must_be_non_empty_string(cls, value: str, info: ValidationInfo) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Field '{info.field_name}' must be a non-empty string, got: {value!r}"
            )
        return value

    @classmethod
    def from_params(cls, params: dict) -> "ModelRegistryConfig":
        return cls(**params)


class MlflowEvaluateConfig(BaseModel):
    target: str = "churn"
    eval_metric: str = "recall_score"
    eval_metric_threshold: float = 0.8
    min_absolute_change: float = 0.05
    min_relative_change: float = 0.05

    @field_validator("target", "eval_metric", mode="before")
    @classmethod
    def must_be_non_empty_string(cls, value: str, info: ValidationInfo) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"'{info.field_name}' must be a non-empty string, got: {value!r}"
            )
        return value

    @field_validator(
        "eval_metric_threshold",
        "min_absolute_change",
        "min_relative_change",
        mode="before",
    )
    @classmethod
    def must_be_valid_ratio(cls, value: float, info: ValidationInfo) -> float:
        if not isinstance(value, (int, float)):
            raise ValueError(f"'{info.field_name}' must be a number, got: {value!r}")
        if not (0.0 <= value <= 1.0):
            raise ValueError(
                f"'{info.field_name}' must be between 0.0 and 1.0, got: {value!r}"
            )
        return value

    @classmethod
    def from_params(cls, params: dict) -> "MlflowEvaluateConfig":
        return cls(**params)
