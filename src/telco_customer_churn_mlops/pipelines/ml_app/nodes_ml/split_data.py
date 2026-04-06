"""
Contains ML nodes of kedro pipelines
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def train_test_split_data(
    feat_df: pd.DataFrame,
    target: pd.Series,
    split_params: dict
) -> tuple:
    """
    Split feature and target data into training and testing sets.

    Args:
        feat_df (pd.DataFrame):
            DataFrame containing feature columns.
        target (pd.Series):
            Series containing the target variable.
        split_params (dict):
            Dictionary containing split configuration:
            - 'test_size' (float): proportion of data to use for testing.
            - 'random_state' (int): random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            - X_train: training features
            - X_test: testing features
            - y_train: training target
            - y_test: testing target
    """
    X_train, X_test, y_train, y_test = train_test_split(
        feat_df,
        target,
        test_size = split_params["test_size"],
        random_state = split_params["random_state"],
        stratify= target
    )

    return X_train, X_test, y_train, y_test
