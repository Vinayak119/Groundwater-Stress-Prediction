"""
explainability.py
=================
SHAP-based model interpretability utilities.

Responsibilities
----------------
- Compute SHAP values for tree-based models (Random Forest, XGBoost, LightGBM).
- Generate SHAP summary plots (beeswarm, bar chart).
- Generate SHAP waterfall plots for individual predictions.
- Generate SHAP dependence plots for specific features.
- Save SHAP outputs to results/shap/.
- Provide helpers for the Streamlit Explainable AI page.

Background
----------
SHAP (SHapley Additive exPlanations) assigns each feature a contribution
value for every prediction. This satisfies the research requirement for
Explainable AI and helps identify which factors drive groundwater stress.

Usage
-----
>>> from src.explainability import compute_shap_values, plot_shap_summary
>>> shap_values, explainer = compute_shap_values(model, X_test, model_type="tree")
>>> fig = plot_shap_summary(shap_values, X_test)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  SHAP value computation                                             #
# ------------------------------------------------------------------ #

def compute_shap_values(
    model,
    X: pd.DataFrame,
    model_type: str = "tree",
    background_data: Optional[pd.DataFrame] = None,
    max_background: int = 100,
):
    """
    Compute SHAP values for a fitted model on feature matrix X.

    Parameters
    ----------
    model : fitted estimator
        Tree-based model (RF, XGBoost, LightGBM) or any sklearn estimator.
    X : pd.DataFrame
        Feature matrix to explain (typically X_test or a representative sample).
    model_type : str, default "tree"
        "tree"    → shap.TreeExplainer  (fast, exact for tree-based models)
        "kernel"  → shap.KernelExplainer (slow, model-agnostic)
    background_data : pd.DataFrame, optional
        Background dataset for KernelExplainer. If None, uses a sample of X.
    max_background : int, default 100
        Maximum background rows for KernelExplainer (keep small for speed).

    Returns
    -------
    shap_values : np.ndarray or shap.Explanation
        SHAP values array, shape (n_samples, n_features).
    explainer : shap.Explainer
        The fitted explainer object (useful for single-instance explanations).
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("Install shap: pip install shap>=0.45") from exc

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
    elif model_type == "kernel":
        bg = background_data if background_data is not None else X.sample(
            min(max_background, len(X)), random_state=42
        )
        explainer = shap.KernelExplainer(model.predict, bg)
        shap_values = explainer.shap_values(X)
    else:
        raise ValueError(f"Unknown model_type='{model_type}'. Use 'tree' or 'kernel'.")

    logger.info("SHAP values computed for %d samples, %d features.", len(X), X.shape[1])
    return shap_values, explainer


# ------------------------------------------------------------------ #
#  Summary plot (global importance)                                   #
# ------------------------------------------------------------------ #

def plot_shap_summary(
    shap_values,
    X: pd.DataFrame,
    plot_type: str = "dot",
    max_display: int = 20,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Generate a SHAP summary plot showing global feature importance.

    Parameters
    ----------
    shap_values : shap.Explanation or np.ndarray
    X : pd.DataFrame
    plot_type : str, default "dot"
        "dot" → beeswarm plot  |  "bar" → mean |SHAP| bar chart
    max_display : int, default 20
        Maximum features to display.
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import shap

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X,
        plot_type=plot_type,
        max_display=max_display,
        show=False,
    )
    fig = plt.gcf()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP summary plot saved to %s", save_path)

    return fig


# ------------------------------------------------------------------ #
#  Waterfall plot (single-instance explanation)                       #
# ------------------------------------------------------------------ #

def plot_shap_waterfall(
    shap_values,
    sample_index: int = 0,
    max_display: int = 15,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Generate a SHAP waterfall plot explaining one prediction.

    Parameters
    ----------
    shap_values : shap.Explanation
        Must be a shap.Explanation object (returned by TreeExplainer(X)).
    sample_index : int, default 0
        Index of the sample to explain.
    max_display : int, default 15
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import shap

    shap.waterfall_plot(shap_values[sample_index], max_display=max_display, show=False)
    fig = plt.gcf()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP waterfall plot saved to %s", save_path)

    return fig


# ------------------------------------------------------------------ #
#  Dependence plot                                                    #
# ------------------------------------------------------------------ #

def plot_shap_dependence(
    shap_values,
    X: pd.DataFrame,
    feature: str,
    interaction_feature: str = "auto",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    SHAP dependence plot: how a single feature affects predictions.

    Parameters
    ----------
    shap_values : shap.Explanation or np.ndarray
    X : pd.DataFrame
    feature : str
        Primary feature to plot on x-axis.
    interaction_feature : str, default "auto"
        Feature for colour encoding. "auto" lets SHAP choose.
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import shap

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        feature,
        shap_values.values if hasattr(shap_values, "values") else shap_values,
        X,
        interaction_index=interaction_feature,
        ax=ax,
        show=False,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP dependence plot saved to %s", save_path)

    return fig


# ------------------------------------------------------------------ #
#  Save SHAP values                                                   #
# ------------------------------------------------------------------ #

def save_shap_values(
    shap_values,
    X: pd.DataFrame,
    path: str | Path = "results/shap/shap_values.parquet",
) -> None:
    """
    Save SHAP values and feature names to a Parquet file for dashboard use.

    Parameters
    ----------
    shap_values : shap.Explanation or np.ndarray
    X : pd.DataFrame
    path : str or Path
    """
    vals = shap_values.values if hasattr(shap_values, "values") else shap_values
    shap_df = pd.DataFrame(vals, columns=X.columns)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    shap_df.to_parquet(path, index=False)
    logger.info("SHAP values saved to %s", path)
