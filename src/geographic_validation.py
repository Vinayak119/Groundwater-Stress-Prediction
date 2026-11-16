"""
geographic_validation.py
========================
Geographic hold-out validation utilities.

Responsibilities
----------------
- Split the dataset into seen-region (train) and unseen-region (hold-out) subsets.
- Persist split definitions as JSON so both team members use identical splits.
- Evaluate a fitted model on the held-out region and return metrics.
- Visualise the geographic distribution of training vs hold-out regions.

Research Question Addressed
---------------------------
"Can a model trained on selected Indian states/districts reliably predict
groundwater levels in geographically unseen regions?"

Design Principles
-----------------
- The hold-out region is defined by state or district name, NOT by random row index.
- The split is fixed before any model training and saved to data/splits/.
- No model sees hold-out data during training or hyperparameter search.

Usage
-----
>>> from src.geographic_validation import make_geo_split, evaluate_on_holdout
>>> train_df, holdout_df = make_geo_split(df, holdout_states=["Rajasthan", "Punjab"])
>>> metrics = evaluate_on_holdout(model, holdout_df, feature_cols, target_col)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Split definition                                                   #
# ------------------------------------------------------------------ #

def make_geo_split(
    df: pd.DataFrame,
    holdout_states: Optional[list[str]] = None,
    holdout_districts: Optional[list[str]] = None,
    split_col: str = "state",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and geographic hold-out subsets.

    Exactly one of `holdout_states` or `holdout_districts` should be provided,
    or use `split_col` to specify an arbitrary column.

    Parameters
    ----------
    df : pd.DataFrame
    holdout_states : list[str], optional
        State names to withhold from training.
    holdout_districts : list[str], optional
        District names to withhold from training.
    split_col : str, default "state"
        Column used to define the geographic unit.

    Returns
    -------
    train_df : pd.DataFrame
        Rows from seen regions.
    holdout_df : pd.DataFrame
        Rows from unseen (hold-out) regions.

    Raises
    ------
    ValueError
        If the split_col is not present in df, or if the hold-out list is empty.
    """
    if split_col not in df.columns:
        raise ValueError(f"Column '{split_col}' not found in DataFrame.")

    holdout_values = holdout_states or holdout_districts
    if not holdout_values:
        raise ValueError("Provide at least one state or district for the hold-out set.")

    holdout_mask = df[split_col].isin(holdout_values)
    train_df = df[~holdout_mask].reset_index(drop=True)
    holdout_df = df[holdout_mask].reset_index(drop=True)

    logger.info(
        "Geographic split: train=%d rows (%d %ss), holdout=%d rows (%d %ss).",
        len(train_df), train_df[split_col].nunique(), split_col,
        len(holdout_df), holdout_df[split_col].nunique(), split_col,
    )
    return train_df, holdout_df


def save_geo_split(
    holdout_values: list[str],
    split_col: str,
    path: str | Path = "data/splits/geo_split.json",
) -> None:
    """
    Persist the geographic split definition as JSON.

    Parameters
    ----------
    holdout_values : list[str]
        The region names forming the hold-out set.
    split_col : str
        The column these names come from (e.g. "state").
    path : str or Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    split_def = {"split_col": split_col, "holdout_values": holdout_values}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split_def, f, indent=2)
    logger.info("Geographic split definition saved to %s", path)


def load_geo_split(path: str | Path = "data/splits/geo_split.json") -> dict:
    """Load a saved geographic split definition."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ #
#  Hold-out evaluation                                                #
# ------------------------------------------------------------------ #

def evaluate_on_holdout(
    model,
    holdout_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_name: str = "model",
) -> dict:
    """
    Generate predictions on the hold-out region and compute metrics.

    Parameters
    ----------
    model : fitted estimator
    holdout_df : pd.DataFrame
    feature_cols : list[str]
    target_col : str
    model_name : str

    Returns
    -------
    dict
        Metric dict as returned by evaluation.compute_metrics().
    """
    from src.evaluation import compute_metrics

    X_holdout = holdout_df[feature_cols]
    y_holdout = holdout_df[target_col]
    y_pred = model.predict(X_holdout)

    return compute_metrics(y_holdout, y_pred, model_name=model_name, split_label="geo_holdout")


def summarise_geo_performance(
    df: pd.DataFrame,
    model,
    feature_cols: list[str],
    target_col: str,
    region_col: str = "state",
    model_name: str = "model",
) -> pd.DataFrame:
    """
    Compute per-region metrics to identify where the model performs best and worst.

    Parameters
    ----------
    df : pd.DataFrame
        Hold-out or full test DataFrame.
    model : fitted estimator
    feature_cols : list[str]
    target_col : str
    region_col : str, default "state"
    model_name : str

    Returns
    -------
    pd.DataFrame
        One row per region with RMSE, MAE, R², n_samples.
    """
    from src.evaluation import compute_metrics

    results = []
    for region, group in df.groupby(region_col):
        if len(group) < 5:
            logger.warning("Skipping region '%s': only %d samples.", region, len(group))
            continue
        y_pred = model.predict(group[feature_cols])
        m = compute_metrics(group[target_col], y_pred, model_name=model_name, split_label=str(region))
        m["region"] = region
        results.append(m)

    return pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
