"""
evaluation.py
=============
Shared evaluation harness for all groundwater-level prediction models.

Responsibilities
----------------
- Compute regression metrics: RMSE, MAE, R².
- Compute the persistence baseline prediction (y_t = y_{t-1}).
- Compare model performance against the persistence baseline.
- Generate residual plots, actual-vs-predicted plots.
- Save metric tables to results/metrics/.
- Support chronological and geographic evaluation splits.

All models (Decision Tree, Random Forest, XGBoost, LightGBM) MUST be
evaluated through this module so their results are directly comparable.

Usage
-----
>>> from src.evaluation import compute_metrics, persistence_baseline, plot_residuals
>>> preds = model.predict(X_test)
>>> metrics = compute_metrics(y_test, preds, model_name="RandomForest")
>>> print(metrics)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Core metric computation                                            #
# ------------------------------------------------------------------ #

def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str = "model",
    split_label: str = "test",
) -> dict[str, float | str]:
    """
    Compute RMSE, MAE, and R² for a set of predictions.

    Parameters
    ----------
    y_true : array-like
        Ground-truth water-level values.
    y_pred : array-like
        Model-predicted water-level values.
    model_name : str
        Label for the model (used in returned dict for identification).
    split_label : str
        Label for the split, e.g. "test", "geo_holdout", "chrono_test".

    Returns
    -------
    dict
        {model_name, split, rmse, mae, r2}
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    metrics = {
        "model": model_name,
        "split": split_label,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "n_samples": len(y_true),
    }
    logger.info(
        "[%s | %s] RMSE=%.4f  MAE=%.4f  R²=%.4f  (n=%d)",
        model_name, split_label, rmse, mae, r2, len(y_true),
    )
    return metrics


def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Convert a list of metric dicts into a sorted comparison DataFrame.

    Parameters
    ----------
    results : list[dict]
        Each dict as returned by compute_metrics().

    Returns
    -------
    pd.DataFrame
        Sorted by RMSE ascending.
    """
    df = pd.DataFrame(results)
    df = df.sort_values("rmse").reset_index(drop=True)
    return df


# ------------------------------------------------------------------ #
#  Persistence baseline                                               #
# ------------------------------------------------------------------ #

def persistence_baseline(
    df: pd.DataFrame,
    well_col: str,
    value_col: str,
    target_col: str,
) -> np.ndarray:
    """
    Compute the persistence baseline predictions.

    The persistence model predicts: ŷ(t) = y(t-1).
    This is the simplest possible benchmark — a model must beat it to
    demonstrate added value from ML.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted chronologically within each well.
    well_col : str
        Well identifier column.
    value_col : str
        Current observed water level (used as the lagged predictor).
    target_col : str
        Actual next-period water level (the target).

    Returns
    -------
    np.ndarray
        Persistence predictions aligned with df rows that have a valid
        previous observation.

    Notes
    -----
    The lag-1 column (if already computed by feature_engineering) is
    equivalent to the persistence prediction and can be used directly.
    """
    df = df.copy().sort_values([well_col, "year", "month"] if "month" in df.columns else [well_col, "year"])
    lag_col = f"{value_col}_lag_1"

    if lag_col in df.columns:
        preds = df[lag_col].values
    else:
        preds = df.groupby(well_col)[value_col].shift(1).values

    logger.info("Persistence baseline computed using lag-1 of '%s'.", value_col)
    return preds


# ------------------------------------------------------------------ #
#  Generalisation gap                                                 #
# ------------------------------------------------------------------ #

def generalisation_gap(
    seen_metrics: dict,
    holdout_metrics: dict,
    metric: str = "rmse",
) -> float:
    """
    Compute the generalisation gap between seen-region and hold-out performance.

    Parameters
    ----------
    seen_metrics : dict
        Metrics on the seen-region test split.
    holdout_metrics : dict
        Metrics on the geographic hold-out split.
    metric : str, default "rmse"
        Which metric to compute the gap for.

    Returns
    -------
    float
        Δ metric (holdout - seen). Positive means worse on unseen regions.
    """
    gap = holdout_metrics[metric] - seen_metrics[metric]
    logger.info(
        "Generalisation gap (%s): seen=%.4f, holdout=%.4f, Δ=%.4f",
        metric, seen_metrics[metric], holdout_metrics[metric], gap,
    )
    return gap


# ------------------------------------------------------------------ #
#  Plotting utilities                                                 #
# ------------------------------------------------------------------ #

def plot_actual_vs_predicted(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str = "Model",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Scatter plot of actual vs predicted water levels.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    model_name : str
    save_path : str or Path, optional
        If provided, saves the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, label="Predictions")
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel("Actual Water Level (m bgl)")
    ax.set_ylabel("Predicted Water Level (m bgl)")
    ax.set_title(f"{model_name}: Actual vs Predicted")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved actual-vs-predicted plot to %s", save_path)

    return fig


def plot_residuals(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str = "Model",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Residual plot (predicted vs residual) to diagnose model bias.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    model_name : str
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.4, s=15)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted Water Level (m bgl)")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title(f"{model_name}: Residual Plot")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved residual plot to %s", save_path)

    return fig


# ------------------------------------------------------------------ #
#  Save metrics to disk                                               #
# ------------------------------------------------------------------ #

def save_metrics(metrics_list: list[dict], path: str | Path) -> None:
    """
    Save a list of metric dicts to a CSV file.

    Parameters
    ----------
    metrics_list : list[dict]
    path : str or Path
        Destination CSV file (e.g. results/metrics/model_comparison.csv).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_list).to_csv(path, index=False)
    logger.info("Metrics saved to %s", path)
