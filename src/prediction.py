"""
prediction.py
=============
Inference helpers for the Streamlit dashboard.

Responsibilities
----------------
- Load a trained model from disk.
- Prepare a single-row or batch feature matrix for inference.
- Return a prediction and optional SHAP explanation for dashboard display.
- Provide a friendly interface that the dashboard can call without
  needing to know about model internals.

Design Principles
-----------------
- The dashboard NEVER re-trains models.
- This module reads from models/trained_models/ and data/processed/.
- Input validation is done here before calling model.predict().

Usage
-----
>>> from src.prediction import load_pipeline, predict_single, predict_batch
>>> pipeline = load_pipeline("models/trained_models/rf_v1.joblib")
>>> result = predict_single(pipeline, well_features)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Pipeline container                                                 #
# ------------------------------------------------------------------ #

@dataclass
class GroundwaterPipeline:
    """
    Container for all artefacts needed to run inference.

    Attributes
    ----------
    model : fitted estimator
    feature_cols : list[str]
        Ordered list of feature column names the model was trained on.
    scaler : optional
        Fitted StandardScaler. If None, features are assumed pre-scaled.
    metadata : dict
        Training metadata (model name, training date, metrics, etc.).
    """
    model: object
    feature_cols: list[str]
    scaler: Optional[object] = None
    metadata: dict = field(default_factory=dict)


def load_pipeline(
    model_path: str | Path,
    scaler_path: Optional[str | Path] = None,
) -> GroundwaterPipeline:
    """
    Load a trained model (and optional scaler) from disk.

    Parameters
    ----------
    model_path : str or Path
        Path to the .joblib model file in models/trained_models/.
    scaler_path : str or Path, optional
        Path to the .joblib scaler file (if features need scaling).

    Returns
    -------
    GroundwaterPipeline
    """
    from src.model_training import load_model, load_model_metadata
    from src.preprocessing import load_scaler

    model = load_model(model_path)
    metadata = load_model_metadata(model_path)
    feature_cols = metadata.get("feature_cols", [])

    scaler = None
    if scaler_path:
        scaler = load_scaler(scaler_path)

    logger.info("Pipeline loaded. Model: %s", metadata.get("model_name", "unknown"))
    return GroundwaterPipeline(model=model, feature_cols=feature_cols, scaler=scaler, metadata=metadata)


# ------------------------------------------------------------------ #
#  Single-row prediction                                              #
# ------------------------------------------------------------------ #

def predict_single(
    pipeline: GroundwaterPipeline,
    features: dict | pd.Series,
) -> dict:
    """
    Predict the next-period water level for a single observation.

    Parameters
    ----------
    pipeline : GroundwaterPipeline
    features : dict or pd.Series
        Feature values keyed by column name.

    Returns
    -------
    dict
        {"predicted_wl_mbgl": float, "model": str}

    Raises
    ------
    ValueError
        If any required feature column is missing.
    """
    row = pd.DataFrame([features]) if isinstance(features, dict) else pd.DataFrame([features])
    _check_features(row, pipeline.feature_cols)
    X = row[pipeline.feature_cols]

    if pipeline.scaler is not None:
        X = pd.DataFrame(pipeline.scaler.transform(X), columns=pipeline.feature_cols)

    pred = float(pipeline.model.predict(X)[0])
    logger.debug("Single prediction: %.4f m bgl", pred)

    return {
        "predicted_wl_mbgl": pred,
        "model": pipeline.metadata.get("model_name", "unknown"),
    }


# ------------------------------------------------------------------ #
#  Batch prediction                                                   #
# ------------------------------------------------------------------ #

def predict_batch(
    pipeline: GroundwaterPipeline,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict water levels for a batch of observations.

    Parameters
    ----------
    pipeline : GroundwaterPipeline
    df : pd.DataFrame
        Must contain all columns in pipeline.feature_cols.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added "predicted_wl_mbgl" column.
    """
    _check_features(df, pipeline.feature_cols)
    X = df[pipeline.feature_cols].copy()

    if pipeline.scaler is not None:
        X = pd.DataFrame(pipeline.scaler.transform(X), columns=pipeline.feature_cols, index=X.index)

    df = df.copy()
    df["predicted_wl_mbgl"] = pipeline.model.predict(X)
    logger.info("Batch prediction complete: %d rows.", len(df))
    return df


# ------------------------------------------------------------------ #
#  Utilities                                                          #
# ------------------------------------------------------------------ #

def _check_features(df: pd.DataFrame, required: list[str]) -> None:
    """Raise ValueError if any required feature is missing from df."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for inference: {missing}")


def describe_pipeline(pipeline: GroundwaterPipeline) -> str:
    """Return a human-readable summary of the loaded pipeline."""
    meta = pipeline.metadata
    lines = [
        f"Model     : {meta.get('model_name', 'Unknown')}",
        f"Features  : {len(pipeline.feature_cols)} columns",
        f"Trained   : {meta.get('trained_at', 'N/A')}",
        f"Train RMSE: {meta.get('train_rmse', 'N/A')}",
        f"Test RMSE : {meta.get('test_rmse', 'N/A')}",
        f"Scaler    : {'Yes' if pipeline.scaler else 'No'}",
    ]
    return "\n".join(lines)
