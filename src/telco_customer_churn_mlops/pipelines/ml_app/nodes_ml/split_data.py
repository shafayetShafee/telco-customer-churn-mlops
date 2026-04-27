import pandas as pd
from sklearn.model_selection import train_test_split

from telco_customer_churn_mlops.configs import SplitConfig


def train_calib_test_split_data(
    feat_df: pd.DataFrame, target: pd.Series, split_params: dict
) -> tuple:
    """
    Split feature and target data into train, calibration, and test sets.

    Parameters
    ----------
    feat_df : pd.DataFrame
        DataFrame containing feature columns.

    target : pd.Series
        Target variable.

    split_params : dict
        Training data split configuration, see SplitConfig for fields.

    Returns
    -------
    X_train : pd.DataFrame
        Training feature set.

    X_calib : pd.DataFrame
        Calibration feature set.

    X_test : pd.DataFrame
        Test feature set.

    y_train : pd.Series
        Training target values.

    y_calib : pd.Series
        Calibration target values.

    y_test : pd.Series
        Test target values.

    Notes
    -----
    Returns a tuple in the following order:
    (X_train, X_calib, X_test, y_train, y_calib, y_test).
    """
    split_cfg = SplitConfig.from_params(split_params)
    test_size = split_cfg.test_size
    calib_size = split_cfg.calib_size
    random_state = split_cfg.random_state

    temp_size = test_size + calib_size

    X_train, X_temp, y_train, y_temp = train_test_split(
        feat_df, target, test_size=temp_size, random_state=random_state, stratify=target
    )

    calib_relative_size = calib_size / temp_size

    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1 - calib_relative_size),
        random_state=random_state,
        stratify=y_temp,
    )

    return X_train, X_calib, X_test, y_train, y_calib, y_test
