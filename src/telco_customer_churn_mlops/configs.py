from pydantic import BaseModel, ValidationInfo, field_validator


class ModelRegistryConfig(BaseModel):
    model_name: str = "calibrated_threshold_classifier"
    champion_alias: str = "champion"
    inference_pipeline_name: str = "churn_inference_pipeline"

    @field_validator('*', mode='before')
    @classmethod
    def must_be_non_empty_string(cls, value: str, info: ValidationInfo) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Field '{info.field_name}' must be a non-empty string, got: {value!r}")
        return value
    
    @classmethod
    def from_params(cls, params: dict) -> "ModelRegistryConfig":
        return cls(**params)
    

