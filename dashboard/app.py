from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.components.ui import setup_page, page_header
from dashboard.components.styles import section_header

root = setup_page(title="Research Dashboard", icon="💧")

st.markdown("""
<div style="padding:1rem 0">
  <div class="hero-title">Groundwater Research Intelligence</div>
  <div class="hero-subtitle">
    Tamil Nadu CGWB observations · Temporal forecasting · Geographic
    generalisation · Model interpretability
  </div>
</div>
""", unsafe_allow_html=True)

def read_csv(relative_path):
    path = root / relative_path
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def read_json(relative_path):
    path = root / relative_path
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

observations = read_csv("data/processed/observations_model_ready.csv")
pairs = read_csv("data/processed/forecasting_pairs_temporally_purged.csv")
benchmark = read_csv("results/metrics/initial_model_benchmark.csv")
geo = read_csv("results/metrics/geographic_holdout_results.csv")
districts = read_csv("results/metrics/geographic_district_performance.csv")
shap = read_csv("results/shap/shap_feature_importance.csv")
intervals = read_csv("results/metrics/preliminary_prediction_intervals.csv")

st.info(
    "Research mode: this dashboard reads project artifacts. "
    "Missing artifacts are reported as unavailable; synthetic demo results "
    "are not substituted."
)

section_header("Research dataset and experiment status")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Clean observations",
        f"{len(observations):,}" if observations is not None else "Unavailable"
    )
with c2:
    st.metric(
        "Forecasting pairs",
        f"{len(pairs):,}" if pairs is not None else "Unavailable"
    )
with c3:
    st.metric(
        "Trained model files",
        str(len(list((root / "models/trained").glob("*.joblib"))))
        if (root / "models/trained").exists() else "0"
    )
with c4:
    st.metric(
        "SHAP features",
        str(len(shap)) if shap is not None else "Unavailable"
    )

st.caption(
    "Observation records are cleaned measurements. Forecasting pairs are "
    "next-observation prediction examples, not fixed 90-day forecasts."
)

section_header("Dataset scope and quality")

if observations is not None:
    date_col = "Parsed_Date"
    value_col = "GWL_m_bgl"
    station_col = "Provisional_Station_ID"

    cols = st.columns(3)
    if date_col in observations.columns:
        dates = pd.to_datetime(observations[date_col], errors="coerce").dropna()
        with cols[0]:
            if not dates.empty:
                st.metric(
                    "Observed date range",
                    f"{dates.min():%Y} – {dates.max():%Y}"
                )
    if station_col in observations.columns:
        with cols[1]:
            st.metric(
                "Provisional station IDs",
                f"{observations[station_col].nunique():,}"
            )
    if value_col in observations.columns:
        values = pd.to_numeric(observations[value_col], errors="coerce")
        with cols[2]:
            st.metric(
                "Missing groundwater values",
                f"{values.isna().sum():,}"
            )

    st.warning(
        "Station identifiers are provisional dataset identifiers, not "
        "verified official well identities. Records with unresolved "
        "identity conflicts and flagged anomalies require careful interpretation."
    )
else:
    st.warning("The cleaned observation CSV is not currently available.")

section_header("Temporal model benchmark")

if benchmark is not None:
    st.dataframe(benchmark, use_container_width=True, hide_index=True)

    st.caption(
        "The initial benchmark is exploratory. On the temporal test split, "
        "the persistence baseline outperformed the initial machine-learning "
        "models on the reported aggregate MAE."
    )
else:
    st.warning("Initial model benchmark file is unavailable.")

section_header("Geographic district holdout")

if geo is not None:
    st.dataframe(geo, use_container_width=True, hide_index=True)

    st.caption(
        "This experiment holds out Tamil Nadu districts. It is not an "
        "independent cross-state validation."
    )
else:
    st.warning("Geographic holdout results are unavailable.")

if districts is not None:
    st.subheader("District-level performance")
    st.dataframe(districts, use_container_width=True, hide_index=True)

section_header("Model explainability")

if shap is not None:
    st.dataframe(shap.head(15), use_container_width=True, hide_index=True)
    st.caption(
        "SHAP values describe model attribution, not causal effects "
        "or proof of hydrogeological mechanisms."
    )
else:
    st.warning("SHAP importance results are unavailable.")

section_header("Prediction uncertainty")

if intervals is not None:
    st.dataframe(intervals, use_container_width=True, hide_index=True)
    st.warning(
        "The current prediction intervals are preliminary. Their calibration "
        "uses validation data involved in model selection and should not yet "
        "be described as final independently calibrated uncertainty."
    )
else:
    st.warning("Preliminary prediction interval results are unavailable.")

section_header("Research interpretation and limitations")

st.markdown("""
- The forecasting target is the next available observation within 150 days.
- Groundwater depth in metres below ground level is not, by itself, an
  official groundwater-stress classification.
- Negative measurements and unusually deep readings are not automatically
  discarded; they require contextual and source verification.
- Provisional station IDs do not establish permanent well identity.
- The temporal test results are exploratory and have already been inspected.
- Geographic results represent district holdout within Tamil Nadu.
- No independent cross-state validation is claimed.
- Model explanations are associations within the fitted model, not causation.
""")

st.caption(
    "Academic research dashboard · CGWB Tamil Nadu groundwater observations"
)
