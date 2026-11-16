"""
preprocessing.py
================
Data cleaning and preprocessing utilities for the CGWB groundwater dataset.

Responsibilities
----------------
- Remove duplicate records.
- Handle missing values using documented strategies.
- Standardise column names and data types.
- Filter wells with insufficient observation history.
- Encode categorical variables (season, aquifer type, state).
- Scale / normalise numerical features (fit on train split ONLY).
- Persist fitted scaler artefacts for dashboard inference.

Design Principles
-----------------
- Scalers and encoders are ALWAYS fit on the training split only.
  Calling transform() on validation/test data uses the already-fitted object.
- No global state: functions accept and return DataFrames explicitly.
- Every cleaning decision is logged so it can be audited.

Usage
-----
>>> from src.preprocessing import clean_cgwb, encode_categoricals, scale_features
>>> df_clean = clean_cgwb(df_raw)
>>> df_encoded, encoder_map = encode_categoricals(df_clean)
>>> df_scaled, scaler = scale_features(df_encoded, feature_cols, fit=True)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Any

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Constants — update after inspecting the actual CGWB raw file.     #
# ------------------------------------------------------------------ #

# Minimum number of observations a well must have to be retained.
# PLACEHOLDER — agree with Manya before finalising.
MIN_OBSERVATIONS_PER_WELL: int = 5

# Categorical columns to be label-encoded.
# PLACEHOLDER — confirm against actual column names.
CATEGORICAL_COLUMNS: list[str] = ["season", "aquifer_type", "state"]

# Numerical columns expected to be non-negative.
NON_NEGATIVE_COLUMNS: list[str] = ["water_level_mbgl"]


# ------------------------------------------------------------------ #
#  Core cleaning pipeline                                             #
# ------------------------------------------------------------------ #

def clean_cgwb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the core data-quality cleaning pipeline to the raw CGWB DataFrame.

    Steps applied (in order):
    1. Standardise column names to snake_case.
    2. Drop fully duplicate rows.
    3. Safely convert numeric observation columns and drop missing values.
    4. Remove physically implausible values (negative water levels).
    5. Reset the index.

    Parameters
    ----------
    df : pd.DataFrame
        Raw CGWB DataFrame as returned by data_loader.load_raw().

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.

    Notes
    -----
    This function does NOT modify the input DataFrame in place.

    Sparse-well filtering is intentionally NOT performed here.
    It is a dataset-scoping decision (how many observations a well must
    have to be included in modelling) that is separate from data quality.
    Call ``filter_sparse_wells()`` explicitly after this function when
    preparing the final modelling dataset.
    """
    df = df.copy()
    if len(df) == 0:
        return df

    n_start = len(df)

    # Step 1: Standardise column names
    df.columns = [_to_snake_case(c) for c in df.columns]
    logger.info("Standardised column names.")

    # Step 2: Drop duplicate rows
    df = df.drop_duplicates()
    logger.info("Dropped %d duplicate rows.", n_start - len(df))

    # Step 3: Drop rows with missing primary observation (safely handle non-numeric values like 'Dry' or '-')
    if "water_level_mbgl" in df.columns:
        df["water_level_mbgl"] = pd.to_numeric(df["water_level_mbgl"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["water_level_mbgl"])
        logger.info("Dropped %d rows with missing or non-numeric water_level_mbgl.", before - len(df))

    # Step 4: Remove implausible values
    for col in NON_NEGATIVE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            before = len(df)
            df = df[df[col] >= 0]
            logger.info("Removed %d rows with negative or invalid %s.", before - len(df), col)

    df = df.reset_index(drop=True)
    logger.info("Cleaning complete. Rows: %d → %d", n_start, len(df))
    return df


def filter_sparse_wells(
    df: pd.DataFrame,
    well_col: str = "well_id",
    min_obs: int = MIN_OBSERVATIONS_PER_WELL,
) -> pd.DataFrame:
    """
    Remove wells that have fewer than ``min_obs`` observations.

    This is a dataset-scoping step — it controls which wells are included
    in the modelling dataset based on how much history they have. It must
    be called explicitly after ``clean_cgwb()``, not inside it, because:

    - The minimum threshold is a research/modelling decision, not a
      data-quality rule.
    - Unit tests for cleaning should not be affected by this threshold.
    - The threshold value (``MIN_OBSERVATIONS_PER_WELL``) may change as
      the dataset is explored.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (output of ``clean_cgwb()``).
    well_col : str, default "well_id"
        Name of the well identifier column.
    min_obs : int, default MIN_OBSERVATIONS_PER_WELL
        Minimum number of rows a well must appear in to be retained.

    Returns
    -------
    pd.DataFrame
        DataFrame with sparse wells removed and index reset.
    """
    if well_col not in df.columns:
        logger.warning(
            "Column '%s' not found — skipping sparse-well filter.", well_col
        )
        return df

    counts = df[well_col].value_counts()
    valid_wells = counts[counts >= min_obs].index
    before = len(df)
    df = df[df[well_col].isin(valid_wells)].reset_index(drop=True)
    logger.info(
        "Filtered wells with <%d observations: %d wells removed, %d rows dropped.",
        min_obs, len(counts) - len(valid_wells), before - len(df),
    )
    return df


# Keep private alias so any internal call sites are not broken silently.
_filter_sparse_wells = filter_sparse_wells


# ------------------------------------------------------------------ #
#  Categorical encoding                                               #
# ------------------------------------------------------------------ #

def encode_categoricals(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Label-encode categorical columns and return the fitted encoders.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str], optional
        Columns to encode. Defaults to CATEGORICAL_COLUMNS.

    Returns
    -------
    df_encoded : pd.DataFrame
        DataFrame with encoded columns (originals replaced).
    encoder_map : dict[str, LabelEncoder]
        Mapping of column name → fitted LabelEncoder.
        Save these to decode predictions in the dashboard.
    """
    df = df.copy()
    columns = columns or CATEGORICAL_COLUMNS
    encoder_map: dict[str, LabelEncoder] = {}

    for col in columns:
        if col not in df.columns:
            logger.warning("Column '%s' not found — skipping encoding.", col)
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoder_map[col] = le
        logger.info("Encoded '%s' → %d classes.", col, len(le.classes_))

    return df, encoder_map


# ------------------------------------------------------------------ #
#  Feature scaling                                                    #
# ------------------------------------------------------------------ #

def scale_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Apply StandardScaler to the specified feature columns.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
        Columns to scale.
    scaler : StandardScaler, optional
        A previously fitted scaler. Required when fit=False.
    fit : bool, default True
        If True, fit a new scaler on df (use for TRAIN split only).
        If False, apply the provided scaler (use for VAL/TEST splits).

    Returns
    -------
    df_scaled : pd.DataFrame
    scaler : StandardScaler
        The fitted scaler (for saving and re-use).

    Raises
    ------
    ValueError
        If fit=False but no scaler is provided.
    """
    if not fit and scaler is None:
        raise ValueError("Must provide a fitted scaler when fit=False.")

    df = df.copy()
    present = [c for c in feature_cols if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[present] = scaler.fit_transform(df[present])
        logger.info("Fitted and applied StandardScaler to %d columns.", len(present))
    else:
        df[present] = scaler.transform(df[present])
        logger.info("Applied existing StandardScaler to %d columns.", len(present))

    return df, scaler


# ------------------------------------------------------------------ #
#  Feature Imputation (Leakage-free)                                 #
# ------------------------------------------------------------------ #

def impute_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    strategy: str = "median",
    imputer: Optional[SimpleImputer] = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, SimpleImputer]:
    """
    Impute missing feature values without future data leakage.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
        Columns to impute.
    strategy : str, default "median"
        Strategy to use for imputation ("median", "mean", "most_frequent").
    imputer : SimpleImputer, optional
        A previously fitted SimpleImputer. Required when fit=False.
    fit : bool, default True
        If True, fit on df (use for TRAIN split only).
        If False, apply provided imputer (use for VAL/TEST splits).

    Returns
    -------
    df_imputed : pd.DataFrame
    imputer : SimpleImputer
        The fitted imputer for saving and transforming unseen splits.
    """
    if not fit and imputer is None:
        raise ValueError("Must provide a fitted imputer when fit=False.")

    df = df.copy()
    present = [c for c in feature_cols if c in df.columns]

    if fit:
        imputer = SimpleImputer(strategy=strategy)
        df[present] = imputer.fit_transform(df[present])
        logger.info("Fitted and applied %s SimpleImputer to %d columns.", strategy, len(present))
    else:
        df[present] = imputer.transform(df[present])
        logger.info("Applied existing SimpleImputer to %d columns.", len(present))

    return df, imputer


# ------------------------------------------------------------------ #
#  Validation helpers for Station & Date Identifiers                  #
# ------------------------------------------------------------------ #

def validate_station_identifiers(
    df: pd.DataFrame,
    well_col: str = "well_id",
) -> dict[str, Any]:
    """
    Validate station identifier consistency and validity.

    Parameters
    ----------
    df : pd.DataFrame
    well_col : str, default "well_id"

    Returns
    -------
    dict[str, Any]
        Validation status and station statistics.
    """
    if well_col not in df.columns:
        return {"is_valid": False, "error": f"Station column '{well_col}' not found."}

    series = df[well_col].astype(str).str.strip()
    n_empty = int((series == "").sum() + df[well_col].isna().sum())
    n_unique = int(series.nunique())

    return {
        "is_valid": n_empty == 0,
        "n_stations": n_unique,
        "n_empty_ids": n_empty,
        "total_records": len(df),
    }


def validate_date_attributes(
    df: pd.DataFrame,
    year_col: str = "year",
    month_col: str = "month",
) -> dict[str, Any]:
    """
    Validate temporal columns (year range, month 1-12).

    Parameters
    ----------
    df : pd.DataFrame
    year_col : str, default "year"
    month_col : str, default "month"

    Returns
    -------
    dict[str, Any]
        Validation status and identified issues.
    """
    issues = []
    if year_col in df.columns and pd.api.types.is_numeric_dtype(df[year_col]):
        years = df[year_col].dropna()
        if (years < 1900).any() or (years > 2050).any():
            issues.append(f"Implausible years detected in '{year_col}'.")
    if month_col in df.columns and pd.api.types.is_numeric_dtype(df[month_col]):
        months = df[month_col].dropna()
        if (months < 1).any() or (months > 12).any():
            issues.append(f"Invalid month values (outside 1-12) in '{month_col}'.")

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
    }


# ------------------------------------------------------------------ #
#  Persistence helpers                                                #
# ------------------------------------------------------------------ #

def save_scaler(scaler: StandardScaler, path: str | Path) -> None:
    """Serialise a fitted scaler to disk using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    logger.info("Scaler saved to %s", path)


def load_scaler(path: str | Path) -> StandardScaler:
    """Load a previously serialised scaler from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scaler file not found: {path}")
    return joblib.load(path)


def save_encoders(encoder_map: dict[str, LabelEncoder], path: str | Path) -> None:
    """
    Serialise categorical label encoders to disk using joblib.

    Parameters
    ----------
    encoder_map : dict[str, LabelEncoder]
        Dictionary mapping column names to fitted LabelEncoder instances.
    path : str or Path
        Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder_map, path)
    logger.info("Categorical encoders saved to %s", path)


def load_encoders(path: str | Path) -> dict[str, LabelEncoder]:
    """
    Load previously serialised categorical label encoders from disk.

    Parameters
    ----------
    path : str or Path
        Source file path.

    Returns
    -------
    dict[str, LabelEncoder]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Encoder file not found: {path}")
    return joblib.load(path)


# ------------------------------------------------------------------ #
#  Utility                                                            #
# ------------------------------------------------------------------ #

def _to_snake_case(name: str) -> str:
    """Convert a column name to lowercase snake_case."""
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )
