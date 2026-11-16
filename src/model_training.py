"""
model_training.py
=================
Model training and serialisation utilities.

Responsibilities
----------------
- Provide a unified train_model() interface for all supported models.
- Enforce the use of fixed random seeds for reproducibility.
- Save trained model artefacts to models/trained_models/.
- Load model artefacts for evaluation and dashboard inference.
- Support the following models (Phase 3):
    * Random Forest Regressor (primary model — Arghya)
    * XGBoost Regressor
    * LightGBM Regressor

Decision Tree is handled by Manya in her notebook (Phase 2).
The persistence baseline is not a trained model — see evaluation.py.

Design Principles
-----------------
- train_model() accepts X_train / y_train only — it never touches
  validation or test data.
- Hyperparameter defaults are stored in MODEL_DEFAULTS and can be
  overridden at call time. Hyperparameter search will be added in Phase 3.
- All serialisation goes through joblib.

Usage
-----
>>> from src.model_training import train_model, save_model, load_model
>>> model = train_model("random_forest", X_train, y_train)
>>> save_model(model, "models/trained_models/rf_v1.joblib")
>>> model = load_model("models/trained_models/rf_v1.joblib")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Fixed random seed — must match across all modules.                #
# ------------------------------------------------------------------ #
RANDOM_SEED: int = 42

# ------------------------------------------------------------------ #
#  Default hyperparameters — to be tuned in Phase 3.                 #
# ------------------------------------------------------------------ #
MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 4,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    },
    "xgboost": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "verbosity": 0,
    },
    "lightgbm": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": -1,
        "num_leaves": 31,
        "random_state": RANDOM_SEED,
        "verbosity": -1,
    },
}

SUPPORTED_MODELS: frozenset[str] = frozenset(MODEL_DEFAULTS.keys())


# ------------------------------------------------------------------ #
#  Core training function                                             #
# ------------------------------------------------------------------ #

def train_model(
    model_name: str,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    hyperparams: dict[str, Any] | None = None,
) -> Any:
    """
    Instantiate, fit, and return a model.

    Parameters
    ----------
    model_name : str
        One of "random_forest", "xgboost", "lightgbm".
    X_train : array-like
        Training features. Must NOT include validation or test rows.
    y_train : array-like
        Training target.
    hyperparams : dict, optional
        Override specific hyperparameters. Merged with MODEL_DEFAULTS.

    Returns
    -------
    Fitted estimator (sklearn-compatible API).

    Raises
    ------
    ValueError
        If model_name is not in SUPPORTED_MODELS.
    ImportError
        If the required library is not installed.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Supported: {sorted(SUPPORTED_MODELS)}"
        )

    params = {**MODEL_DEFAULTS[model_name], **(hyperparams or {})}

    model = _instantiate_model(model_name, params)
    logger.info("Training '%s' with params: %s", model_name, params)
    model.fit(X_train, y_train)
    logger.info("Training complete.")
    return model


def _instantiate_model(model_name: str, params: dict[str, Any]) -> Any:
    """Return a fresh (unfitted) estimator for the given model name."""
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**params)

    if model_name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(**params)

    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**params)

    raise ValueError(f"No instantiation logic for '{model_name}'.")


# ------------------------------------------------------------------ #
#  Serialisation helpers                                              #
# ------------------------------------------------------------------ #

def save_model(model: Any, path: str | Path, metadata: dict | None = None) -> None:
    """
    Serialise a fitted model to disk.

    Parameters
    ----------
    model : fitted estimator
    path : str or Path
        Destination file (should end in .joblib).
    metadata : dict, optional
        Optional metadata dict stored alongside the model (e.g. feature names,
        training date, metric scores). Saved as a companion .meta.joblib file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)

    if metadata:
        meta_path = path.with_suffix(".meta.joblib")
        joblib.dump(metadata, meta_path)
        logger.info("Model metadata saved to %s", meta_path)


def load_model(path: str | Path) -> Any:
    """
    Load a serialised model from disk.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    Fitted estimator.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return model


def load_model_metadata(path: str | Path) -> dict:
    """Load companion metadata for a serialised model."""
    meta_path = Path(path).with_suffix(".meta.joblib")
    if not meta_path.exists():
        return {}
    return joblib.load(meta_path)


# ------------------------------------------------------------------ #
#  Feature importance helper (works for tree-based models)           #
# ------------------------------------------------------------------ #

def get_feature_importances(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """
    Extract feature importances from a tree-based model.

    Parameters
    ----------
    model : fitted estimator with .feature_importances_ attribute
    feature_names : list[str]

    Returns
    -------
    pd.DataFrame
        Sorted by importance descending, columns: [feature, importance].
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Model does not expose feature_importances_.")

    return (
        pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
