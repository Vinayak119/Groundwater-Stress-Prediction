"""
feature_engineering.py
======================
Feature construction for the groundwater-level prediction pipeline.

Responsibilities
----------------
- Create temporal lag features (water level at t-1, t-2, …).
- Compute rolling statistics (mean, std, min, max over a window).
- Encode seasons and months as cyclical (sine/cosine) features.
- Compute year-over-year change features.
- Generate the prediction target (next-period water level).
- Ensure NO data leakage: all features at time t use only t or earlier.

Design Principles
-----------------
- Features are built per-well using groupby to avoid cross-well leakage.
- The function that generates the target variable is kept SEPARATE from
  the function that generates input features.
- All window sizes are explicit parameters — never hardcoded inside logic.

Usage
-----
>>> from src.feature_engineering import build_lag_features, build_target, build_cyclical_features
>>> df = build_lag_features(df, well_col="well_id", value_col="water_level_mbgl", lags=[1, 2, 3])
>>> df = build_target(df, well_col="well_id", value_col="water_level_mbgl", horizon=1)
>>> df = build_cyclical_features(df, month_col="month")
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Lag features                                                       #
# ------------------------------------------------------------------ #

def build_lag_features(
    df: pd.DataFrame,
    well_col: str,
    value_col: str,
    lags: list[int],
    sort_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Create lagged copies of `value_col` for each lag in `lags`,
    computed within each well group to prevent cross-well leakage.

    Parameters
    ----------
    df : pd.DataFrame
    well_col : str
        Column identifying the well (groupby key).
    value_col : str
        The column to lag (e.g. "water_level_mbgl").
    lags : list[int]
        Positive integers specifying lag periods. E.g. [1, 2, 3].
    sort_cols : list[str], optional
        Columns to sort by before computing lags.
        Defaults to ["year", "month"] if present.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with new columns named f"{value_col}_lag_{k}".

    Notes
    -----
    Rows where lagged values are unavailable (start of each well's series)
    will contain NaN. These rows are typically dropped before modelling.
    """
    df = df.copy()
    if sort_cols is None:
        sort_cols = [c for c in ["year", "month"] if c in df.columns]

    if sort_cols:
        df = df.sort_values([well_col] + sort_cols).reset_index(drop=True)

    for lag in lags:
        col_name = f"{value_col}_lag_{lag}"
        df[col_name] = df.groupby(well_col)[value_col].shift(lag)
        logger.debug("Created lag feature: %s", col_name)

    logger.info("Built %d lag features for '%s'.", len(lags), value_col)
    return df


# ------------------------------------------------------------------ #
#  Rolling statistics                                                 #
# ------------------------------------------------------------------ #

def build_rolling_features(
    df: pd.DataFrame,
    well_col: str,
    value_col: str,
    windows: list[int],
    stats: Optional[list[str]] = None,
    sort_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute rolling window statistics per well without cross-well leakage.

    Parameters
    ----------
    df : pd.DataFrame
    well_col : str
    value_col : str
    windows : list[int]
        Window sizes (number of periods). E.g. [3, 6, 12].
    stats : list[str], optional
        Statistics to compute. Defaults to ["mean", "std", "min", "max"].
    sort_cols : list[str], optional
        Columns to sort by before computing rolling windows.
        Defaults to ["year", "month"] if present.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with new rolling stat columns.
    """
    df = df.copy()
    if sort_cols is None:
        sort_cols = [c for c in ["year", "month"] if c in df.columns]

    if sort_cols:
        df = df.sort_values([well_col] + sort_cols).reset_index(drop=True)

    stats = stats or ["mean", "std", "min", "max"]

    for window in windows:
        for stat in stats:
            col_name = f"{value_col}_roll_{window}_{stat}"
            # Station-wise transformation guarantees no values cross well boundaries
            df[col_name] = df.groupby(well_col)[value_col].transform(
                lambda s: getattr(s.shift(1).rolling(window, min_periods=1), stat)()
            )
            logger.debug("Created rolling feature: %s", col_name)

    logger.info(
        "Built rolling features for windows=%s, stats=%s on '%s'.",
        windows, stats, value_col,
    )
    return df


# ------------------------------------------------------------------ #
#  Year-over-year change                                              #
# ------------------------------------------------------------------ #

def build_yoy_change(
    df: pd.DataFrame,
    well_col: str,
    value_col: str,
    sort_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute year-over-year change in water level per well.

    The change is computed as: value(t) - value(t-1 year).
    Uses diff(1) within well groups after chronological sorting.

    Parameters
    ----------
    df : pd.DataFrame
    well_col : str
    value_col : str
    sort_cols : list[str], optional
        Columns to sort by before computing diff.

    Returns
    -------
    pd.DataFrame
        With new column f"{value_col}_yoy_change".
    """
    df = df.copy()
    if sort_cols is None:
        sort_cols = [c for c in ["year", "month"] if c in df.columns]

    if sort_cols:
        df = df.sort_values([well_col] + sort_cols).reset_index(drop=True)

    col_name = f"{value_col}_yoy_change"
    df[col_name] = df.groupby(well_col)[value_col].diff(1)
    logger.info("Created YoY change feature: %s", col_name)
    return df


# ------------------------------------------------------------------ #
#  Cyclical temporal encoding                                         #
# ------------------------------------------------------------------ #

def build_cyclical_features(
    df: pd.DataFrame,
    month_col: str = "month",
    season_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Encode month (and optionally season) as sine/cosine pairs
    so the model understands cyclical temporal structure.

    Parameters
    ----------
    df : pd.DataFrame
    month_col : str, default "month"
        Integer month column (1–12).
    season_col : str, optional
        If provided, also encode this as a cyclical variable
        (assumes integer encoding 0–3 for 4 seasons).

    Returns
    -------
    pd.DataFrame
        With new columns: month_sin, month_cos (and season_sin/cos if applicable).
    """
    df = df.copy()

    if month_col in df.columns:
        df["month_sin"] = np.sin(2 * math.pi * df[month_col] / 12)
        df["month_cos"] = np.cos(2 * math.pi * df[month_col] / 12)
        logger.info("Encoded '%s' as cyclical (sin/cos).", month_col)

    if season_col and season_col in df.columns:
        n_seasons = df[season_col].nunique()
        df["season_sin"] = np.sin(2 * math.pi * df[season_col] / n_seasons)
        df["season_cos"] = np.cos(2 * math.pi * df[season_col] / n_seasons)
        logger.info("Encoded '%s' as cyclical (sin/cos).", season_col)

    return df


# ------------------------------------------------------------------ #
#  Target variable generation                                         #
# ------------------------------------------------------------------ #

def build_target(
    df: pd.DataFrame,
    well_col: str,
    value_col: str,
    horizon: int = 1,
    target_col: str = "target_wl_next",
    sort_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Generate the prediction target: water level `horizon` periods ahead.

    Uses groupby + shift(-horizon) so each row's target is the future
    value within the same well, without leaking across wells.

    Parameters
    ----------
    df : pd.DataFrame
    well_col : str
    value_col : str
        Source column for the future value.
    horizon : int, default 1
        Number of periods ahead to predict.
    target_col : str, default "target_wl_next"
        Name of the target column.
    sort_cols : list[str], optional
        Columns to sort by before computing target.
        Defaults to ["year", "month"] if present.

    Returns
    -------
    pd.DataFrame
        With the target column appended.

    Notes
    -----
    The last `horizon` rows of each well's series will have NaN targets
    and must be dropped before training.
    """
    df = df.copy()
    if sort_cols is None:
        sort_cols = [c for c in ["year", "month"] if c in df.columns]

    if sort_cols:
        df = df.sort_values([well_col] + sort_cols).reset_index(drop=True)

    df[target_col] = df.groupby(well_col)[value_col].shift(-horizon)
    n_nan = df[target_col].isna().sum()
    logger.info(
        "Built target '%s' (horizon=%d). NaN rows from end-of-series: %d.",
        target_col, horizon, n_nan,
    )
    return df


# ------------------------------------------------------------------ #
#  Convenience: drop NaN rows after feature/target construction       #
# ------------------------------------------------------------------ #

def drop_incomplete_rows(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Drop any rows that have NaN in feature or target columns.

    Call this AFTER all lag/rolling/target features have been computed.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
        Columns to check for NaN (should include the target column).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with no NaN in the specified columns.
    """
    before = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    logger.info("Dropped %d rows with NaN in feature/target columns.", before - len(df))
    return df
