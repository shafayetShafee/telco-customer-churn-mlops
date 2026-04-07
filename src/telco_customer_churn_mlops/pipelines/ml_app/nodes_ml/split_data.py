"""
Contains ML nodes of kedro pipelines
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def train_calib_test_split_data(
    feat_df: pd.DataFrame,
    target: pd.Series,
    split_params: dict
) -> tuple:
    """
    Split feature and target data into train, calibration, and test sets.

    Args:
        feat_df (pd.DataFrame):
            DataFrame containing feature columns.
        target (pd.Series):
            Series containing the target variable.
        split_params (dict):
            Dictionary containing split configuration:
            - 'test_size' (float): proportion of full data for test set.
            - 'calib_size' (float): proportion of full data for calibration set.
            - 'random_state' (int): random seed for reproducibility.

    Returns:
        Tuple:
            - X_train, X_calib, X_test
            - y_train, y_calib, y_test
    """

    test_size = split_params["test_size"]
    calib_size = split_params["calib_size"]
    random_state = split_params["random_state"]

    temp_size = test_size + calib_size

    X_train, X_temp, y_train, y_temp = train_test_split(
        feat_df,
        target,
        test_size=temp_size,
        random_state=random_state,
        stratify=target
    )

    calib_relative_size = calib_size / temp_size

    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1 - calib_relative_size),
        random_state=random_state,
        stratify=y_temp
    )

    return X_train, X_calib, X_test, y_train, y_calib, y_test
