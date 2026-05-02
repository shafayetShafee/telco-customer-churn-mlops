import logging
from typing import Literal

import great_expectations as gx
import pandas as pd
from great_expectations.core import ExpectationSuiteValidationResult
from great_expectations.expectations import (
    ExpectColumnValuesToBeOfType,
    ExpectTableColumnsToMatchSet,
)

logger = logging.getLogger(__name__)


def build_expectation_suite(
    name: str,
    target_column: str,
    features: list[str],
    column_types: dict,
    data_type: str,
) -> gx.ExpectationSuite:
    """
    Dynamically builds a Great Expectations suite for a given dataset.

    Parameters
    ----------
    name : str
        Dataset name, used to label the suite as ``{name}_validation``.

    target_column : str
        Name of the target column. Included only when ``data_type="train"``.

    features : list[str]
        Feature column names expected in the dataset.

    column_types : dict[str, str]
        Mapping of column name to expected pandas dtype.

    data_type : {"train", "infer"}
        Controls whether the target column is included in validation.

    Returns
    -------
    gx.ExpectationSuite

    Raises
    ------
    ValueError
        If ``data_type`` is not ``"train"`` or ``"infer"``.
    """
    if data_type == "train":
        expected_columns = features + [target_column]
    elif data_type == "infer":
        expected_columns = features
    else:
        raise ValueError(
            f"Invalid data_type: '{data_type}'. Must be 'train' or 'infer'."
        )

    suite = gx.ExpectationSuite(name=f"{name}_validation")
    suite.expectations = [
        ExpectTableColumnsToMatchSet(column_set=expected_columns, exact_match=False),
        *[
            ExpectColumnValuesToBeOfType(column=col, type_=column_types[col])
            for col in expected_columns
        ],
    ]

    return suite


def validate_dataset(  # noqa
    df: pd.DataFrame,
    features: list[str],
    column_types: dict,
    data_type: Literal["train", "infer"],
    dataset_name: str,
    target_column: str | None = None,
) -> ExpectationSuiteValidationResult:
    """
    Validate a dataset against a Great Expectations suite.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate.

    target_column : str
        Name of the target column.

    features : list[str]
        Feature column names expected in the dataset.

    column_types : dict
        Mapping of column name to expected pandas dtype.

    data_type : {"train", "infer"}
        Controls whether the target column is included in validation.

    dataset_name : str
        Name used to register the data source and label validation errors.

    Raises
    ------
    ValueError
        If validation fails or ``data_type`` is invalid.
    """
    context = gx.get_context()
    source = context.data_sources.add_or_update_pandas(dataset_name)
    asset = source.add_dataframe_asset(dataset_name)
    batch_request = asset.build_batch_request(options={"dataframe": df})
    batch = asset.get_batch(batch_request)

    suite = build_expectation_suite(
        name=dataset_name,
        target_column=target_column,
        features=features,
        column_types=column_types,
        data_type=data_type,
    )

    result = batch.validate(suite)

    if not result.success:
        failed = [r for r in result.results if not r.success]
        for r in failed:
            logger.warning(
                "FAILED: %s \n %s \n %s",
                r.expectation_config.type,
                r.expectation_config.get("kwargs"),
                r.get("result"),
            )

    stats = result.statistics
    logger.info(
        "GX Validation: Evaluated epectations: %d | Passed: %d | Failed: %d | Success percent: %.1f%%",
        stats["evaluated_expectations"],
        stats["successful_expectations"],
        stats["unsuccessful_expectations"],
        stats["success_percent"],
    )

    return result
