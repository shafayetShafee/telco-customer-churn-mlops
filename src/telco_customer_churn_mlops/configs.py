from typing import Annotated, Literal, Self, Optional

from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    model_validator,
)


def check_empty_string(value: str) -> str:
    """
    Validate that a string is not empty or whitespace-only after stripping.

    Parameters
    ----------
    value : str
        The string value to validate.

    Returns
    -------
    str
        The original string if valid.

    Raises
    ------
    ValueError
        If the string is empty or contains only whitespace.
    """
    if not value.strip():
        raise ValueError(f"must be a non-empty string, got: {value!r}")
    return value


NonEmptyStr = Annotated[str, AfterValidator(check_empty_string), Field(strict=True)]
ValidRatio = Annotated[float, Field(gt=0, lt=1, strict=True)]


class BaseConfig(BaseModel):
    @classmethod
    def from_params(cls, params: dict) -> Self:
        return cls(**params)


class SplitConfig(BaseConfig):
    test_size: ValidRatio = 0.15
    calib_size: ValidRatio = 0.15
    random_state: Annotated[int, Field(ge=0, strict=True)] = 42

    @model_validator(mode="after")
    def splits_must_not_exceed_one(self) -> Self:
        total = self.test_size + self.calib_size
        if total >= 1.0:
            raise ValueError(
                f"test_size ({self.test_size}) + calib_size ({self.calib_size}) "
                f"= {total:.2f} — must be less than 1.0."
            )
        return self


class OptunaConfig(BaseConfig):
    cv: Annotated[int, Field(ge=2, le=10, strict=True)] = 2
    n_trials: Annotated[int, Field(ge=2, strict=True)] = 2
    random_state: Annotated[int, Field(ge=0, strict=True)] = 42


class PrefitCalibConfig(BaseConfig):
    cv: Literal["prefit"] = "prefit"
    inductive: Annotated[bool, Field(strict=True)] = False
    random_state: Annotated[int, Field(ge=0, strict=True)] = 42


class ShapConfig(BaseConfig):
    explainer_name: NonEmptyStr = "shap_explainer"
    bg_data_size: Annotated[int, Field(ge=1, strict=True)] = 1000
    eval_data_size: Annotated[int | None, Field(ge=1, strict=True)] = None
    plots_max_display: Annotated[int, Field(ge=1, le=30, strict=True)] = 25
    random_state: Annotated[int, Field(ge=0, strict=True)] = 42


class MlflowEvaluateConfig(BaseConfig):
    target: NonEmptyStr = "churn"
    eval_metric: NonEmptyStr = "recall_score"
    eval_metric_threshold: ValidRatio = 0.80
    min_absolute_change: ValidRatio = 0.05
    min_relative_change: ValidRatio = 0.05


class ModelRegistryConfig(BaseConfig):
    model_name: NonEmptyStr = "calibrated_threshold_classifier"
    champion_alias: NonEmptyStr = "champion"
    inference_pipeline_name: NonEmptyStr = "churn_inference_pipeline"
    always_replace: Annotated[bool, Field(strict=True)] = False
    eval_metric: NonEmptyStr = "recall_score"


class InferenceConfig(BaseConfig):
    prediction_col: NonEmptyStr = "churn_prediction"
    proba_col: NonEmptyStr = "churn_probability"
