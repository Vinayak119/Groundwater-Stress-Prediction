"""
data_loader.py
==============
Utilities for loading raw and processed CGWB groundwater datasets.

Responsibilities
----------------
- Locate and read the raw CGWB Excel / CSV source files from data/raw/.
- Load the processed Parquet file produced by the preprocessing pipeline.
- Validate that required columns are present before returning a DataFrame.
- Verify unit and measurement-reference compatibility (MBGL vs MSL).
- Provide a lightweight schema-check against the agreed column contract.

Usage
-----
>>> from src.data_loader import load_raw, load_processed, check_measurement_compatibility
>>> df_raw = load_raw("data/raw/cgwb_data.csv")
>>> df = load_processed("data/processed/cgwb_processed.parquet")
>>> audit = check_measurement_compatibility(df)

Notes
-----
- This module is READ-ONLY with respect to data files.
- Never modify source files from within this module.
- All paths should be passed as arguments; avoid hardcoded absolute paths.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Column contract — must be satisfied by the processed dataset.      #
#  Primary Dataset: Tamil Nadu CGWB groundwater data, 1991–2020.     #
# ------------------------------------------------------------------ #
REQUIRED_PROCESSED_COLUMNS: list[str] = [
    # Identifiers
    "well_id",
    "state",
    "district",
    # Spatial
    "latitude",
    "longitude",
    # Temporal
    "year",
    "month",
    "season",
    # Observation (Strictly in Metres Below Ground Level — MBGL)
    "water_level_mbgl",
    # Target (to be finalized upon research agreement — e.g. target_wl_next)
]


def load_raw(filepath: str | Path) -> pd.DataFrame:
    """
    Load a raw CGWB data file (CSV or Excel) into a DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to the raw data file inside data/raw/.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with no preprocessing applied.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file extension is not supported (.csv, .xlsx, .xls).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif suffix in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(path)
        except ImportError as exc:
            raise ImportError(
                f"Reading Excel files ({suffix}) requires 'openpyxl'. "
                "Install it via: pip install openpyxl"
            ) from exc
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .csv or .xlsx.")

    logger.info("Loaded raw data: %s rows, %s cols from %s", len(df), len(df.columns), path.name)
    return df


def load_processed(
    filepath: str | Path,
    validate: bool = True,
    verify_units: bool = True,
) -> pd.DataFrame:
    """
    Load the processed Parquet dataset produced by Manya's preprocessing pipeline.

    Parameters
    ----------
    filepath : str or Path
        Path to the .parquet file inside data/processed/.
    validate : bool, default True
        If True, checks that all required columns are present.
    verify_units : bool, default True
        If True, verifies that water levels are in MBGL and not MSL.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame ready for feature engineering / modelling.

    Raises
    ------
    FileNotFoundError
        If the Parquet file does not exist.
    ValueError
        If required columns are missing or if unit incompatibility is detected.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Processed data file not found: {path}")

    df = pd.read_parquet(path)
    logger.info("Loaded processed data: %s rows, %s cols from %s", len(df), len(df.columns), path.name)

    if validate:
        _validate_columns(df, REQUIRED_PROCESSED_COLUMNS)

    if verify_units:
        compat = check_measurement_compatibility(df)
        if not compat["is_compatible"]:
            raise ValueError(f"Measurement unit incompatibility detected: {compat['warnings']}")

    return df


def _validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """
    Assert that all required columns are present in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    required : list[str]
        Column names that must exist.

    Raises
    ------
    ValueError
        Lists every missing column so the error is actionable.
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Processed dataset is missing {len(missing)} required column(s): {missing}\n"
            "Check that Manya's preprocessing pipeline has been run and the schema is up to date."
        )
    logger.debug("Column validation passed for %s columns.", len(required))


def check_measurement_compatibility(df: pd.DataFrame) -> dict[str, Any]:
    """
    Verify that groundwater measurements use consistent reference units (MBGL vs MSL).

    CRITICAL HYDROGEOLOGICAL PRINCIPLE:
    - MBGL (Metres Below Ground Level): Depth to water from ground surface (positive values, typically 0–100 m).
    - MSL / AMSL (Above Mean Sea Level): Elevation head relative to sea level.
    These units are NOT interchangeable without surveyed wellhead ground elevations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing groundwater observation columns.

    Returns
    -------
    dict
        Audit report containing:
        - unit_detected: 'MBGL', 'MSL', or 'UNKNOWN'
        - is_compatible: bool
        - warnings: list[str]
        - summary: str
    """
    if len(df) == 0:
        return {
            "unit_detected": "EMPTY",
            "vertical_reference": "N/A (Empty DataFrame)",
            "is_compatible": False,
            "warnings": ["DataFrame is empty."],
            "summary": "Unit audit: Empty DataFrame provided.",
        }

    col_names_lower = [str(c).lower() for c in df.columns]
    warnings: list[str] = []
    unit_detected = "UNKNOWN"
    vertical_reference = "UNKNOWN"
    is_compatible = True

    # 1. Column name check
    has_mbgl_col = any("mbgl" in c for c in col_names_lower)
    has_msl_col = any("msl" in c or "amsl" in c for c in col_names_lower)

    if has_mbgl_col and has_msl_col:
        warnings.append(
            "CRITICAL: Both MBGL and MSL column names detected in the same dataset. "
            "They must not be merged or compared without surveyed well elevation conversion."
        )
        is_compatible = False
        unit_detected = "MIXED"
        vertical_reference = "Mixed (Depth below ground + Elevation head)"
    elif has_msl_col:
        warnings.append(
            "MSL (Above Mean Sea Level) detected. The primary pipeline requires depth to water in "
            "MBGL (Metres Below Ground Level). Conversion requires well surface elevation (Z_surface - WL_msl)."
        )
        is_compatible = False
        unit_detected = "MSL"
        vertical_reference = "MSL/AMSL (Elevation above mean sea level)"
    elif has_mbgl_col or "water_level_mbgl" in df.columns:
        unit_detected = "MBGL"
        vertical_reference = "MBGL (Depth below ground level)"

    # 2. Value range sanity check for MBGL
    target_col = None
    for cand in ["water_level_mbgl", "water_level", "wl_mbgl"]:
        if cand in df.columns:
            target_col = cand
            break

    if target_col is not None and pd.api.types.is_numeric_dtype(df[target_col]):
        vals = df[target_col].dropna()
        if len(vals) > 0:
            if (vals < 0).any():
                neg_count = int((vals < 0).sum())
                warnings.append(
                    f"Found {neg_count} negative water level values in '{target_col}'. "
                    "Physical MBGL depth cannot be negative unless artesian/overflowing."
                )
            # If values exceed plausible groundwater depth (e.g. >200m) or cluster in hundreds of meters,
            # it might be elevation AMSL rather than MBGL
            if vals.median() > 150.0:
                warnings.append(
                    f"Median value of '{target_col}' is {vals.median():.1f}m. "
                    "This high value strongly suggests elevation Above Mean Sea Level (MSL) rather than depth below ground (MBGL)."
                )
                is_compatible = False
                unit_detected = "SUSPECTED_MSL"
                vertical_reference = "Suspected MSL/AMSL (Elevated Piezometric Head)"

    summary = (
        f"Unit audit: Detected {unit_detected}. "
        f"{'Compatible with MBGL pipeline.' if is_compatible else 'Incompatible with primary pipeline specification.'}"
    )

    return {
        "unit_detected": unit_detected,
        "vertical_reference": vertical_reference,
        "is_compatible": is_compatible,
        "warnings": warnings,
        "summary": summary,
    }


def get_schema_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a concise summary of column names, dtypes and null counts.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Summary table with columns: [column, dtype, null_count, null_pct].
    """
    summary = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.values,
        "null_count": df.isnull().sum().values,
        "null_pct": (df.isnull().mean() * 100).round(2).values,
    })
    return summary


# ------------------------------------------------------------------ #
#  Dataset schema & unit comparison tool                              #
# ------------------------------------------------------------------ #

def compare_dataset_schemas(
    primary_df: pd.DataFrame,
    secondary_df: pd.DataFrame,
    primary_label: str = "Tamil Nadu (1991–2020)",
    secondary_label: str = "Tamil Nadu (2021–2025)",
) -> dict[str, Any]:
    """
    Compare two CGWB datasets to rigorously evaluate compatibility.

    Inspects:
    1. Column name alignment (shared, primary-only, secondary-only).
    2. Data types for shared columns.
    3. Measurement unit compatibility (MBGL vs MSL / AMSL).
    4. Well identifier overlap and format consistency.
    5. Temporal attributes and date/year ranges.
    6. Missingness rate comparison.

    Parameters
    ----------
    primary_df : pd.DataFrame
        Reference primary dataset (e.g. Tamil Nadu 1991–2020).
    secondary_df : pd.DataFrame
        Candidate or conditional dataset (e.g. Tamil Nadu 2021–2025 or holdout).
    primary_label : str, default "Tamil Nadu (1991–2020)"
    secondary_label : str, default "Tamil Nadu (2021–2025)"

    Returns
    -------
    dict[str, Any]
        Detailed diagnostic report containing compatibility verdict, flags,
        and comparison tables.
    """
    if len(primary_df) == 0 or len(secondary_df) == 0:
        empty_which = []
        if len(primary_df) == 0:
            empty_which.append(primary_label)
        if len(secondary_df) == 0:
            empty_which.append(secondary_label)
        return {
            "is_compatible": False,
            "primary_label": primary_label,
            "secondary_label": secondary_label,
            "primary_shape": list(primary_df.shape),
            "secondary_shape": list(secondary_df.shape),
            "shared_columns": [],
            "primary_only_columns": list(primary_df.columns),
            "secondary_only_columns": list(secondary_df.columns),
            "type_mismatches": [],
            "primary_unit_audit": check_measurement_compatibility(primary_df),
            "secondary_unit_audit": check_measurement_compatibility(secondary_df),
            "well_id_analysis": {"well_id_present_both": False},
            "temporal_analysis": {},
            "missingness_comparison": [],
            "blocking_reasons": [f"Empty dataset partition provided: {', '.join(empty_which)}."],
        }

    p_cols = set(primary_df.columns)
    s_cols = set(secondary_df.columns)

    shared_cols = sorted(list(p_cols.intersection(s_cols)))
    p_only_cols = sorted(list(p_cols - s_cols))
    s_only_cols = sorted(list(s_cols - p_cols))

    # 1. Type compatibility across shared columns
    type_mismatches = []
    for col in shared_cols:
        p_dtype = str(primary_df[col].dtype)
        s_dtype = str(secondary_df[col].dtype)
        if p_dtype != s_dtype:
            type_mismatches.append({
                "column": col,
                f"{primary_label}_dtype": p_dtype,
                f"{secondary_label}_dtype": s_dtype,
            })

    # 2. Unit audit for both datasets
    p_unit_audit = check_measurement_compatibility(primary_df)
    s_unit_audit = check_measurement_compatibility(secondary_df)

    # 3. Well ID overlap & format
    well_id_analysis: dict[str, Any] = {"well_id_present_both": False}
    p_well_col = next((c for c in ["well_id", "station_code", "site_name"] if c in primary_df.columns), None)
    s_well_col = next((c for c in ["well_id", "station_code", "site_name"] if c in secondary_df.columns), None)

    if p_well_col and s_well_col:
        p_wells = set(primary_df[p_well_col].astype(str).unique())
        s_wells = set(secondary_df[s_well_col].astype(str).unique())
        overlap = p_wells.intersection(s_wells)
        well_id_analysis = {
            "well_id_present_both": True,
            "primary_col": p_well_col,
            "secondary_col": s_well_col,
            "primary_distinct_wells": len(p_wells),
            "secondary_distinct_wells": len(s_wells),
            "common_wells": len(overlap),
            "overlap_ratio_secondary": round(len(overlap) / len(s_wells), 4) if len(s_wells) > 0 else 0.0,
        }

    # 4. Temporal inspection
    temporal_analysis: dict[str, Any] = {}
    p_year_col = next((c for c in ["year", "observation_year"] if c in primary_df.columns), None)
    s_year_col = next((c for c in ["year", "observation_year"] if c in secondary_df.columns), None)
    if p_year_col and pd.api.types.is_numeric_dtype(primary_df[p_year_col]):
        temporal_analysis[f"{primary_label}_years"] = [
            int(primary_df[p_year_col].min()),
            int(primary_df[p_year_col].max()),
        ]
    if s_year_col and pd.api.types.is_numeric_dtype(secondary_df[s_year_col]):
        temporal_analysis[f"{secondary_label}_years"] = [
            int(secondary_df[s_year_col].min()),
            int(secondary_df[s_year_col].max()),
        ]

    # 5. Missingness comparison across shared columns
    missingness = []
    for col in shared_cols:
        p_pct = round(float(primary_df[col].isnull().mean() * 100), 2)
        s_pct = round(float(secondary_df[col].isnull().mean() * 100), 2)
        missingness.append({
            "column": col,
            f"{primary_label}_null_pct": p_pct,
            f"{secondary_label}_null_pct": s_pct,
            "delta_pct": round(s_pct - p_pct, 2),
        })

    # Overall compatibility evaluation
    blocking_reasons: list[str] = []
    if not s_unit_audit["is_compatible"]:
        blocking_reasons.append(
            f"Measurement unit incompatibility in {secondary_label}: {s_unit_audit['summary']}"
        )
    if not p_unit_audit["is_compatible"]:
        blocking_reasons.append(
            f"Measurement unit incompatibility in {primary_label}: {p_unit_audit['summary']}"
        )
    if len(type_mismatches) > 0:
        blocking_reasons.append(
            f"{len(type_mismatches)} shared columns have conflicting data types."
        )
    if len(shared_cols) == 0:
        blocking_reasons.append("Zero shared columns found between the two datasets.")

    # Check key primary columns missing in secondary
    for req in REQUIRED_PROCESSED_COLUMNS:
        if req in p_cols and req not in s_cols:
            blocking_reasons.append(f"Required primary column '{req}' missing in {secondary_label}.")

    is_compatible = (len(blocking_reasons) == 0)

    return {
        "is_compatible": is_compatible,
        "primary_label": primary_label,
        "secondary_label": secondary_label,
        "primary_shape": list(primary_df.shape),
        "secondary_shape": list(secondary_df.shape),
        "shared_columns": shared_cols,
        "primary_only_columns": p_only_cols,
        "secondary_only_columns": s_only_cols,
        "type_mismatches": type_mismatches,
        "primary_unit_audit": p_unit_audit,
        "secondary_unit_audit": s_unit_audit,
        "well_id_analysis": well_id_analysis,
        "temporal_analysis": temporal_analysis,
        "missingness_comparison": missingness,
        "blocking_reasons": blocking_reasons,
    }
