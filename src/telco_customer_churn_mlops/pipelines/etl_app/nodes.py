"""
Contains Data Preprocessing Nodes (functions) of the
Kedro Pipelines
"""

import re
import unicodedata

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def clean_names(  # noqa: PLR0913
    df: pd.DataFrame,
    case: str = "snake",
    remove_special: bool = True,
    strip_accents: bool = True,
    strip_underscores: str | bool = "both",
    truncate: int | None = None,
) -> pd.DataFrame:
    """
    Clean column names in a pandas DataFrame (janitor-style).

    Parameters
    ----------
    case : {"snake", "lower", "upper", "preserve"}
    remove_special : remove non-alphanumeric characters
    strip_accents : remove unicode accents
    strip_underscores : {"left", "right", "both", True, None}
    truncate : int, optional max length

    Returns
    -------
    pd.DataFrame
    """

    def _clean(col: str) -> str:
        col = str(col)

        if strip_accents:
            col = "".join(
                c
                for c in unicodedata.normalize("NFD", col)
                if not unicodedata.combining(c)
            )

        col = re.sub(r"[ /:,?()\.\-]+", "_", col)
        col = re.sub(r"['’]", "", col)
        col = re.sub(r"\xa0", "_", col)
        col = re.sub(r"(\w)@(\w)", r"\1_\2", col)

        # --- case handling ---
        if case == "snake":
            col = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", col)
            col = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", col)
            col = col.lower()
        elif case == "lower":
            col = col.lower()
        elif case == "upper":
            col = col.upper()
        elif case == "preserve":
            pass
        else:
            raise ValueError("case must be one of: snake, lower, upper, preserve")

        if remove_special:
            col = re.sub(r"[^a-zA-Z0-9_]", "", col)

        col = re.sub(r"_+", "_", col)

        if strip_underscores in {"left", "l"}:
            col = col.lstrip("_")
        elif strip_underscores in {"right", "r"}:
            col = col.rstrip("_")
        elif strip_underscores in {"both", True}:
            col = col.strip("_")

        if truncate:
            col = col[:truncate]

        return col

    return df.rename(columns=_clean)


def extract_encode_target_col(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Extract and encode a binary target column from a DataFrame.

    This function selects the specified target column from the input
    DataFrame and encodes its values using a fixed mapping:
    'Yes' -> 1 and 'No' -> 0.

    Args:
        df (pd.DataFrame):
            Input DataFrame containing the target column.
        target_col (str):
            Name of the target column to extract and encode.

    Returns:
        pd.Series:
            Encoded target column as numeric values (1 and 0).
    """
    y_enc = df[target_col].map({"Yes": 1, "No": 0}).rename(target_col.lower())

    return y_enc


def extract_preprocess_features_data(
    df: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """
    Extract and preprocess feature columns from a DataFrame.

    This function selects the specified feature columns and applies a series
    of preprocessing steps, including:
    - Encoding binary categorical variables into numeric values
    - Handling special categorical cases (e.g., 'No phone service', 'No internet service')
    - Creating derived features from existing columns
    - Converting data types for memory efficiency
    - One-hot encoding selected multi-category features
    - Cleaning column names

    The transformations are designed to prepare the dataset for machine
    learning models within a Kedro pipeline.

    Args:
        df (pd.DataFrame):
            Input DataFrame containing raw feature data.
        features (List[str]):
            List of column names to be selected and processed.

    Returns:
        pd.DataFrame:
            Preprocessed feature DataFrame with numeric and encoded values,
            ready for model training or inference.
    """
    feat_df = df[features]

    binary_cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    ]
    internet_service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    pre_proc_df = (
        feat_df.pipe(
            lambda df_: df_.assign(
                **{
                    c: df_[c]
                    .map({"Yes": 1, "No": 0, "Male": 1, "Female": 0})
                    .astype("float")
                    for c in binary_cols
                }
            )
        )
        .assign(
            multiple_lines=lambda df_: (
                df_["MultipleLines"]
                .replace({"No phone service": "No"})
                .map({"Yes": 1, "No": 0})
                .astype("float")
            ),
            no_internet_service=lambda df_: (df_["InternetService"] == "No").astype(
                "float"
            ),
            dsl_internet_service=lambda df_: (df_["InternetService"] == "DSL").astype(
                "float"
            ),
        )
        .pipe(
            lambda df_: df_.assign(
                **{
                    c: df_[c]
                    .replace({"No internet service": "No"})
                    .map({"Yes": 1, "No": 0})
                    for c in internet_service_cols
                }
            )
        )
        .assign(
            total_charges=lambda df_: pd.to_numeric(
                df_["TotalCharges"], errors="coerce"
            )
        )
        .drop(columns=["MultipleLines", "InternetService", "TotalCharges"])
        .pipe(
            lambda df_: df_.assign(
                **{c: df_[c].astype("float") for c in df_.select_dtypes("bool").columns}
            )
        )
    )
    return pre_proc_df


def fit_multi_cat_encoder(df: pd.DataFrame, multi_cat_cols: list[str]) -> OneHotEncoder:
    """
    Fit a OneHotEncoder on multiple categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing categorical features.

    multi_cat_cols : list of str
        List of column names in `df` to be one-hot encoded.

    Returns
    -------
    OneHotEncoder
        Fitted OneHotEncoder instance configured with:
        - drop='first' to avoid multicollinearity
        - dtype='float'
        - handle_unknown='ignore' to safely transform unseen categories
        - sparse_output=False to return dense arrays

    Notes
    -----
    The encoder is fitted only on the specified categorical columns.
    This function does not modify the input dataframe.
    """
    enc = OneHotEncoder(
        drop="first", dtype="float", handle_unknown="ignore", sparse_output=False
    )
    enc.fit(df[multi_cat_cols])
    return enc


def apply_multi_cat_encoder(
    df: pd.DataFrame, encoder: OneHotEncoder, multi_cat_cols: list[str]
) -> pd.DataFrame:
    """
    Apply a fitted OneHotEncoder to transform categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing categorical features to transform.

    encoder : OneHotEncoder
        A previously fitted OneHotEncoder instance.

    multi_cat_cols : list of str
        List of column names in `df` to be one-hot encoded.

    Returns
    -------
    pd.DataFrame
        A new dataframe where:
        - Original categorical columns are removed
        - One-hot encoded columns are appended
        - Index is preserved from the input dataframe

    Notes
    -----
    - Unseen categories during transformation are ignored due to
      `handle_unknown='ignore'`.
    - Output columns follow the naming convention from
      `encoder.get_feature_names_out`.
    - This function does not mutate the input dataframe.
    """
    encoded = encoder.transform(df[multi_cat_cols])
    cols = encoder.get_feature_names_out(multi_cat_cols)
    encoded_df = pd.DataFrame(encoded, columns=cols, index=df.index)
    return df.drop(columns=multi_cat_cols).join(encoded_df).astype('float')
