from __future__ import annotations
from typing import List, Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


# =========================================================
# Basic utilities
# =========================================================

def drop_unnecessary_columns(
    df: pd.DataFrame,
    columns_to_drop: List[str],
) -> pd.DataFrame:
    """
    Drop specified columns if they exist.
    """
    return df.drop(columns=columns_to_drop, errors="ignore").copy()


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/validation split.
    """
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )


def separate_inputs_targets(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features (X) and target (y).
    """
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y


def identify_column_types(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical columns.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols


# =========================================================
# Categorical encoding
# =========================================================

def encode_categorical_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """
    One-Hot encode categorical features.
    """

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
        drop="if_binary",
    )

    encoder.fit(train_df[categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    train_encoded = pd.DataFrame(
        encoder.transform(train_df[categorical_cols]),
        columns=encoded_cols,
        index=train_df.index,
    )

    val_encoded = pd.DataFrame(
        encoder.transform(val_df[categorical_cols]),
        columns=encoded_cols,
        index=val_df.index,
    )

    train_df = pd.concat(
        [train_df.drop(columns=categorical_cols), train_encoded],
        axis=1,
    )

    val_df = pd.concat(
        [val_df.drop(columns=categorical_cols), val_encoded],
        axis=1,
    )

    return train_df, val_df, encoder


# =========================================================
# Numeric scaling
# =========================================================

def scale_numeric_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    numeric_cols: List[str],
    scaler_numeric: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[MinMaxScaler]]:
    """
    Optionally scale numeric features.
    """

    if not scaler_numeric:
        return train_df, val_df, None

    scaler = MinMaxScaler()
    scaler.fit(train_df[numeric_cols])

    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])

    return train_df, val_df, scaler


# =========================================================
# MAIN FUNCTION
# =========================================================

def preprocess_data(
    raw_df: pd.DataFrame,
    target_col: str = "Exited",
    scaler_numeric: bool = False,
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    List[str],
    Optional[MinMaxScaler],
    OneHotEncoder,
]:
    """
    Full preprocessing pipeline for Bank Churn dataset.

    Steps:
    1. Drop Surname and CustomerId
    2. Stratified train/validation split
    3. One-Hot encoding
    4. Optional scaling

    Returns:
        X_train,
        train_targets,
        X_val,
        val_targets,
        input_cols,
        scaler,
        encoder
    """

    df = raw_df.copy()

    # ✅ Remove useless columns
    df = drop_unnecessary_columns(df, ["Surname", "CustomerId"])

    # Split data
    train_df, val_df = split_data(df, target_col)

    X_train, train_targets = separate_inputs_targets(train_df, target_col)
    X_val, val_targets = separate_inputs_targets(val_df, target_col)

    numeric_cols, categorical_cols = identify_column_types(X_train)

    # Encode categorical
    X_train, X_val, encoder = encode_categorical_features(
        X_train,
        X_val,
        categorical_cols,
    )

    # Scale numeric (optional)
    X_train, X_val, scaler = scale_numeric_features(
        X_train,
        X_val,
        numeric_cols,
        scaler_numeric,
    )

    input_cols = X_train.columns.tolist()

    return (
        X_train,
        train_targets,
        X_val,
        val_targets,
        input_cols,
        scaler,
        encoder,
    )


# =========================================================
# NEW DATA
# =========================================================

def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    encoder: OneHotEncoder,
    scaler: Optional[MinMaxScaler] = None,
    scaler_numeric: bool = False,
) -> pd.DataFrame:
    """
    Preprocess new/test data using trained encoder and scaler.
    """

    df = new_df.copy()

    # Same columns removed as during training
    df = drop_unnecessary_columns(df, ["Surname", "CustomerId"])

    numeric_cols, categorical_cols = identify_column_types(df)

    # Encode categorical
    encoded = encoder.transform(df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoded_cols,
        index=df.index,
    )

    df = pd.concat(
        [df.drop(columns=categorical_cols), encoded_df],
        axis=1,
    )

    # Scale numeric (optional)
    if scaler_numeric and scaler is not None:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Ensure identical features as training
    df = df.reindex(columns=input_cols, fill_value=0)

    return df
