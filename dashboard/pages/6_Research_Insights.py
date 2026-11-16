from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.ui import setup_page, page_header
from dashboard.components.styles import PLOTLY_TEMPLATE

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
METRICS = ROOT / "results" / "metrics"
SHAP = ROOT / "results" / "shap"
REPORTS = ROOT / "reports"

FILES = {
    "cleaned observations": DATA / "observations_model_ready.csv",
    "quarantined observations": DATA / "observations_quarantined.csv",
    "forecasting pairs": DATA / "forecasting_pairs_next_observation_150d.csv",
    "purged forecasting pairs": DATA / "forecasting_pairs_temporally_purged.csv",
    "temporal benchmark": METRICS / "initial_model_benchmark.csv",
    "geographic holdout": METRICS / "geographic_holdout_results.csv",
    "district results": METRICS / "geographic_district_performance.csv",
    "gap results": METRICS / "test_performance_by_gap.csv",
    "SHAP importance": SHAP / "shap_feature_importance.csv",
    "prediction intervals": METRICS / "preliminary_prediction_intervals.csv",
}


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def safe_read(name: str) -> pd.DataFrame | None:
    path = FILES[name]
    if not path.exists():
        return None
    try:
        return read_csv(str(path))
    except Exception as exc:
        st.warning(f"Could not read {name}: {exc}")
        return None


def numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def fmt_count(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError, OverflowError):
        return "Unavailable"


def fmt_metric(value, digits: int = 3, suffix: str = "") -> str:
    try:
        number = float(value)
        if np.isfinite(number):
            return f"{number:,.{digits}f}{suffix}"
    except (TypeError, ValueError):
        pass
    return "Unavailable"


def kpi(label: str, value: str, detail: str, accent: str = "violet") -> None:
    st.html(
        f"""
        <div class="research-kpi research-kpi-{accent}">
            <div class="research-kpi-label">{label}</div>
            <div class="research-kpi-value">{value}</div>
            <div class="research-kpi-detail">{detail}</div>
        </div>
        """
    )


def section_note(text: str) -> None:
    st.markdown(
        f'<div class="research-note">{text}</div>',
        unsafe_allow_html=True,
    )


setup_page("Research Insights", "Research Summary")
st.title("Research Insights")
st.caption(
    "A clear summary of the groundwater data, model results, validation, "
    "feature importance and current project limitations."
)

st.html(
    """
    <style>
    .main .block-container {max-width: 1360px; padding-top: 1.5rem; padding-bottom: 3rem;}
    .research-hero {
        background:
          radial-gradient(circle at 90% 10%,rgba(167,139,250,.19),transparent 34%),
          radial-gradient(circle at 5% 100%,rgba(52,211,153,.08),transparent 32%),
          linear-gradient(120deg,#242039,#151522 75%);
        border:1px solid #39304F;border-radius:17px;
        padding:1.15rem 1.35rem;margin:.35rem 0 1.25rem;
        color:#E9E6F3;line-height:1.65;font-size:.88rem;
    }
    .research-kpi {
        background:linear-gradient(145deg,#242039,#181824);
        border:1px solid #39304F;border-radius:13px;
        padding:.9rem 1rem;min-height:108px;margin-bottom:.65rem;
    }
    .research-kpi-label {color:#B4B4C2;font-size:.68rem;font-weight:750;
        letter-spacing:.08em;text-transform:uppercase;}
    .research-kpi-value {color:#F8FAFC;font-size:1.55rem;font-weight:800;
        letter-spacing:-.04em;margin-top:.4rem;overflow-wrap:anywhere;}
    .research-kpi-detail {color:#A1A1AA;font-size:.75rem;line-height:1.45;margin-top:.22rem;}
    .research-note {color:#A1A1AA;font-size:.82rem;line-height:1.6;margin:-.35rem 0 .85rem;}
    .research-panel {background:#181824;border:1px solid #39304F;border-radius:13px;
        padding:1rem 1.1rem;margin:.35rem 0 .75rem;}
    .research-panel-title {color:#F8FAFC;font-weight:750;font-size:.92rem;}
    .research-panel-text {color:#A1A1AA;font-size:.77rem;line-height:1.55;margin-top:.3rem;}
    .research-caveat {background:rgba(120,78,28,.18);border:1px solid rgba(245,190,101,.35);
        border-left:4px solid #F4BE65;border-radius:10px;padding:.9rem 1.05rem;
        color:#FDE7B0;font-size:.82rem;line-height:1.65;margin:.65rem 0;}
    .research-positive {background:rgba(52,211,153,.08);border:1px solid rgba(52,211,153,.25);
        border-left:4px solid #34D399;border-radius:10px;padding:.9rem 1.05rem;
        color:#D1FAE5;font-size:.82rem;line-height:1.65;margin:.65rem 0;}
    </style>
    """
)

st.html(
    """
    <div class="research-hero">
      <strong>Project overview · Tamil Nadu groundwater</strong><br>
      Use this page to understand what data went into the project, how the
      prediction models performed, where they were tested, and what their
      results do—and do not—tell us. All reported values come from the
      project's saved results; unfinished work and limitations are identified.
    </div>
    """
)

# Load each artifact once for consistent page-wide use.
data = {name: safe_read(name) for name in FILES}


# ---------------------------------------------------------------------
# RESEARCH STATUS AND DATA SNAPSHOT
# ---------------------------------------------------------------------
st.subheader("Research status")
status_items = list(FILES.items())
status_cols = st.columns(5)
for col, (name, path) in zip(status_cols, status_items[:5]):
    with col:
        kpi(
            name.title(),
            "Available" if path.exists() else "Missing",
            "Project artifact",
            "green" if path.exists() else "amber",
        )

missing = [name for name, path in FILES.items() if not path.exists()]
if missing:
    st.warning("Some research artefacts are not present: " + ", ".join(missing))

clean = data["cleaned observations"]
quarantine = data["quarantined observations"]
pairs = data["purged forecasting pairs"]

clean_count = len(clean) if clean is not None else None
quarantine_count = len(quarantine) if quarantine is not None else None
pair_count = len(pairs) if pairs is not None else None

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi("Clean observations", fmt_count(clean_count), "Retained groundwater measurements", "violet")
with c2:
    kpi("Quarantined records", fmt_count(quarantine_count), "Held aside for identity or quality review", "amber")
with c3:
    kpi("Purged forecast pairs", fmt_count(pair_count), "Next-observation pairs after temporal purge", "green")
with c4:
    station_count = (
        clean["Provisional_Station_ID"].nunique()
        if clean is not None and "Provisional_Station_ID" in clean.columns
        else None
    )
    kpi("Provisional stations", fmt_count(station_count), "Not verified official well identities", "blue")


# ---------------------------------------------------------------------
# RESEARCH QUESTIONS
# ---------------------------------------------------------------------
st.divider()
st.subheader("1 · Research questions and evidence")
section_note("The status below distinguishes completed analyses from work that has not yet been evaluated.")

research_questions = [
    (
        "RQ1 · Temporal forecasting performance",
        "Do machine-learning regressors improve on a persistence baseline when predicting the next available groundwater observation?",
        "Completed initial temporal benchmark. Persistence achieved lower overall test MAE than all four initial machine-learning models.",
        "Completed",
    ),
    (
        "RQ2 · Geographic generalisation within Tamil Nadu",
        "How does LightGBM perform on districts withheld from model training?",
        "Completed seven-district geographic holdout. LightGBM outperformed persistence overall on this evaluation, while district-level errors varied.",
        "Completed",
    ),
    (
        "RQ3 · Model interpretability",
        "Which input features contribute most strongly to LightGBM predictions?",
        "Completed global SHAP analysis on 1,000 validation observations. Current groundwater level, historical mean and current month had the largest mean absolute SHAP contributions.",
        "Completed",
    ),
    (
        "RQ4 · Forecast-gap sensitivity",
        "Does model performance change according to the time until the next available observation?",
        "Completed four forecast-gap comparisons. Persistence had lower MAE in the first three groups; LightGBM had marginally lower MAE in the final group but higher RMSE.",
        "Completed",
    ),
    (
        "RQ5 · Cross-state transfer",
        "Can a model trained in Tamil Nadu generalise to a geographically unseen neighbouring state?",
        "Not yet evaluated. No Karnataka or Andhra Pradesh external-state validation result is available in the current project artefacts.",
        "Future work",
    ),
]
for title, question, evidence, status in research_questions:
    with st.expander(f"{title}  ·  {status}", expanded=False):
        st.markdown("**Research question**")
        st.write(question)
        st.markdown("**Current evidence**")
        st.write(evidence)


# ---------------------------------------------------------------------
# DATA AUDIT
# ---------------------------------------------------------------------
st.divider()
st.subheader("2 · Data sources, cleaning and audit")
section_note("The project uses CGWB manual groundwater-level observations for Tamil Nadu, combining historical and recent source files.")

overview_cols = st.columns(3)
with overview_cols[0]:
    kpi("Districts represented", "33", "In cleaned observation data", "blue")
with overview_cols[1]:
    kpi("Coordinate gaps", "246", "Records cannot support coordinate-based mapping", "amber")
with overview_cols[2]:
    kpi("RL_MSL", "Largely missing", "Not used as a general model predictor", "violet")

st.markdown(
    """
    **Observed data characteristics**
    - 77,024 cleaned observations.
    - 2,318 provisional station identifiers.
    - Historical observed dates begin in January 1996; the recent source extends through January 2025.
    - Records with missing coordinates may contribute to temporal modelling but cannot support coordinate-based mapping or geographic grouping.

    **Data-quality decisions**
    - Repeated provisional-station/calendar-date conflicts were quarantined rather than averaged.
    - Maruthur2 records were quarantined pending confirmation of official well identity.
    - Unusual high and negative groundwater-level readings were retained for review rather than automatically deleted.
    - Negative readings may represent real conditions and require interpretation using measurement context.
    - Provisional station identifiers must not be assumed to be verified official well identities.
    """
)

with st.expander("Inspect available data-audit artefacts"):
    audit_files = [
        DATA / "observations_model_ready.csv",
        DATA / "observations_quarantined.csv",
        REPORTS / "calendar_date_conflict_groups.csv",
    ]
    audit_table = pd.DataFrame(
        {
            "Artifact": [path.name for path in audit_files],
            "Status": ["Available" if path.exists() else "Not found" for path in audit_files],
        }
    )
    st.dataframe(audit_table, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------
# FORECASTING METHODOLOGY
# ---------------------------------------------------------------------
st.divider()
st.subheader("3 · Forecasting methodology and temporal split")
m1, m2 = st.columns([1.1, 1])
with m1:
    st.markdown(
        """
        **Prediction target:** the next available groundwater-level observation
        within a maximum 150-day gap. This is not a fixed 90-day or quarterly forecast.

        **Target variable:** `Target_GWL_m_bgl`.

        **Model inputs (12 features):**
        - Current groundwater level.
        - Previous and two-observations-back groundwater levels.
        - Historical mean and standard deviation.
        - Number of observations in station history.
        - Days since the previous observation.
        - Current month and sine/cosine month encodings.
        - Latitude and longitude.

        Historical features are constructed from observations at or before the
        forecast origin. The future target date and target gap are not model inputs.
        """
    )
with m2:
    split_rows = [
        {"Split": "Training", "Rule": "Target date through 2018-12-31", "Rows": 51816},
        {"Split": "Validation", "Rule": "Input from 2019-01-01; target through 2020-12-31", "Rows": 3227},
        {"Split": "Test", "Rule": "Input date from 2021-01-01 onward", "Rows": 8350},
        {"Split": "Boundary exclusions", "Rule": "Pairs removed to prevent split overlap", "Rows": 1041},
    ]
    split_df = pd.DataFrame(split_rows)
    split_display = split_df.copy()
    split_display["Rows"] = split_display["Rows"].map(fmt_count)
    st.dataframe(split_display, hide_index=True, use_container_width=True)

st.html(
    """
    <div class="research-caveat">
      <strong>Validation caveat.</strong> The test split uses forecast origins from
      2021 onward and may use earlier observations from the same station as
      historical context. This is a prospective forecasting setup, not a fully
      independent station or external-state test. Test results have also been
      inspected during exploratory model development.
    </div>
    """
)


# ---------------------------------------------------------------------
# TEMPORAL BENCHMARK
# ---------------------------------------------------------------------
st.divider()
st.subheader("4 · Temporal benchmark findings")
benchmark = data["temporal benchmark"]
if benchmark is not None and not benchmark.empty:
    benchmark = numeric_frame(benchmark, ["MAE", "RMSE", "R2"])
    if {"Model", "Split", "MAE", "RMSE"}.issubset(benchmark.columns):
        split_text = benchmark["Split"].astype(str).str.lower()
        test = benchmark[split_text == "test"].copy()
        validation = benchmark[split_text == "validation"].copy()

        if not test.empty and not validation.empty:
            best_val = validation.loc[validation["MAE"].idxmin()]
            best_test = test.loc[test["MAE"].idxmin()]
            persistence = test[test["Model"].astype(str).str.lower() == "persistence baseline"]
            lightgbm = test[test["Model"].astype(str).str.lower() == "lightgbm"]

            a, b, c = st.columns(3)
            with a:
                kpi("Best validation MAE", f'{best_val["MAE"]:.3f} m', str(best_val["Model"]), "violet")
            with b:
                kpi("Best test MAE", f'{best_test["MAE"]:.3f} m', str(best_test["Model"]), "green")
            with c:
                if not persistence.empty:
                    kpi("Persistence test MAE", f'{persistence.iloc[0]["MAE"]:.3f} m', "Temporal test baseline", "blue")
                else:
                    kpi("Persistence test MAE", "Unavailable", "Baseline row not found", "amber")

            chart_l, chart_r = st.columns(2)
            with chart_l:
                mae_fig = px.bar(
                    benchmark,
                    x="Model",
                    y="MAE",
                    color="Split",
                    barmode="group",
                    color_discrete_map={"validation": "#A78BFA", "test": "#34D399"},
                    template=PLOTLY_TEMPLATE,
                    text_auto=".2f",
                    title="MAE · validation vs test",
                )
                mae_fig.update_layout(height=360, xaxis_title="", yaxis_title="MAE (m)", margin=dict(t=55, b=90))
                mae_fig.update_xaxes(tickangle=-18)
                st.plotly_chart(mae_fig, use_container_width=True)
                st.caption(
                    "Lower MAE means predictions are closer to observed groundwater "
                    "levels on average. Compare validation and test bars to see how "
                    "well performance carries over to later dates."
                )
            with chart_r:
                rmse_fig = px.bar(
                    benchmark,
                    x="Model",
                    y="RMSE",
                    color="Split",
                    barmode="group",
                    color_discrete_map={"validation": "#A78BFA", "test": "#34D399"},
                    template=PLOTLY_TEMPLATE,
                    text_auto=".2f",
                    title="RMSE · validation vs test",
                )
                rmse_fig.update_layout(height=360, xaxis_title="", yaxis_title="RMSE (m)", margin=dict(t=55, b=90))
                rmse_fig.update_xaxes(tickangle=-18)
                st.plotly_chart(rmse_fig, use_container_width=True)

            if not persistence.empty and not lightgbm.empty:
                p_mae = float(persistence.iloc[0]["MAE"])
                l_mae = float(lightgbm.iloc[0]["MAE"])
                if p_mae > 0:
                    difference = (l_mae - p_mae) / p_mae * 100
                    st.html(
                        f"""
                        <div class="research-caveat">
                          <strong>Temporal test result:</strong> persistence had
                          {abs(difference):.1f}% {"lower" if difference > 0 else "higher"}
                          MAE than LightGBM on this test split. The initial machine-learning
                          models therefore did not beat persistence on aggregate test MAE.
                        </div>
                        """
                    )
        else:
            st.warning("Temporal benchmark does not contain both validation and test rows.")

        with st.expander("View full temporal benchmark table"):
            st.dataframe(benchmark, hide_index=True, use_container_width=True)
    else:
        st.warning("Temporal benchmark is missing one or more expected columns: Model, Split, MAE, RMSE.")
else:
    st.warning("Temporal benchmark CSV is unavailable.")


# ---------------------------------------------------------------------
# GEOGRAPHIC HOLDOUT AND DISTRICT RESULTS
# ---------------------------------------------------------------------
st.divider()
st.subheader("5 · Geographic holdout findings")
st.caption(
    "This experiment tests the model on selected districts held out from training. "
    "It provides a practical check of geographic generalisation within Tamil Nadu."
)
geo = data["geographic holdout"]
if geo is not None and not geo.empty:
    geo = numeric_frame(geo, ["MAE", "RMSE", "R2"])
    st.dataframe(geo, hide_index=True, use_container_width=True)

    if {"Model", "MAE", "RMSE"}.issubset(geo.columns):
        model_names = geo["Model"].astype(str).str.lower()
        p_rows = geo[model_names.str.contains("persistence", na=False)]
        l_rows = geo[model_names.str.contains("lightgbm", na=False)]
        if not p_rows.empty and not l_rows.empty:
            p_row, l_row = p_rows.iloc[0], l_rows.iloc[0]
            mae_gain = (p_row["MAE"] - l_row["MAE"]) / p_row["MAE"] * 100 if p_row["MAE"] else np.nan
            rmse_gain = (p_row["RMSE"] - l_row["RMSE"]) / p_row["RMSE"] * 100 if p_row["RMSE"] else np.nan
            g1, g2 = st.columns(2)
            with g1:
                kpi("LightGBM geographic MAE", f'{l_row["MAE"]:.3f} m', f'{mae_gain:.1f}% lower than persistence', "green")
            with g2:
                kpi("LightGBM geographic RMSE", f'{l_row["RMSE"]:.3f} m', f'{rmse_gain:.1f}% lower than persistence', "blue")
else:
    st.warning("Geographic holdout results are unavailable.")

district = data["district results"]
if district is not None and not district.empty:
    st.markdown("**District-level LightGBM results**")
    district = numeric_frame(district, ["MAE", "RMSE", "R2", "Stations", "Predictions"])
    if {"District", "RMSE"}.issubset(district.columns):
        district_sorted = district.sort_values("RMSE", ascending=True)
        chart = px.bar(
            district_sorted,
            x="RMSE",
            y="District",
            orientation="h",
            color="RMSE",
            color_continuous_scale=["#34D399", "#A78BFA", "#F59E0B"],
            template=PLOTLY_TEMPLATE,
            text_auto=".2f",
            title="District-wise RMSE · lower is better",
        )
        chart.update_layout(height=360, yaxis_title="", xaxis_title="RMSE (m)", coloraxis_showscale=False, margin=dict(t=55, b=35))
        st.plotly_chart(chart, use_container_width=True)
        st.caption(
            "A higher district RMSE means larger prediction errors in that district. "
            "The chart highlights where model performance may need closer review."
        )
        worst = district.loc[district["RMSE"].idxmax()]
        best = district.loc[district["RMSE"].idxmin()]
        d1, d2 = st.columns(2)
        with d1:
            kpi("Lowest district RMSE", f'{best["RMSE"]:.3f} m', str(best["District"]), "green")
        with d2:
            kpi("Highest district RMSE", f'{worst["RMSE"]:.3f} m', str(worst["District"]), "amber")
    with st.expander("View district-level results table"):
        st.dataframe(district, hide_index=True, use_container_width=True)

st.html(
    """
    <div class="research-caveat">
      This is an internal Tamil Nadu district holdout, not a cross-state transfer
      test or independent external validation. District sample sizes differ, and
      high errors should be investigated against station history and source records.
    </div>
    """
)


# ---------------------------------------------------------------------
# FORECAST GAP ANALYSIS
# ---------------------------------------------------------------------
st.divider()
st.subheader("6 · Forecast-gap sensitivity")
gap = data["gap results"]
if gap is not None and not gap.empty:
    st.caption(
        "The project compared performance across four forecast-gap groups. "
        "The table below displays the stored result fields without assuming "
        "unverified column meanings."
    )
    st.dataframe(gap, hide_index=True, use_container_width=True)
else:
    st.info("Forecast-gap results artefact is not available.")


# ---------------------------------------------------------------------
# SHAP EXPLAINABILITY
# ---------------------------------------------------------------------
st.divider()
st.subheader("7 · Explainability findings")
shap_df = data["SHAP importance"]
if shap_df is not None and not shap_df.empty:
    shap_df = numeric_frame(shap_df, ["Mean_Absolute_SHAP", "Importance_Percent"])
    if {"Feature", "Mean_Absolute_SHAP", "Importance_Percent"}.issubset(shap_df.columns):
        shap_df = shap_df.sort_values("Mean_Absolute_SHAP", ascending=False)
        top = shap_df.head(10).sort_values("Mean_Absolute_SHAP")
        top["Feature label"] = top["Feature"].astype(str).str.replace("_", " ", regex=False)
        shap_fig = px.bar(
            top,
            x="Importance_Percent",
            y="Feature label",
            orientation="h",
            template=PLOTLY_TEMPLATE,
            color="Importance_Percent",
            color_continuous_scale=["#4C3B73", "#A78BFA", "#34D399"],
            text_auto=".1f",
            title="Top global SHAP features",
        )
        shap_fig.update_layout(height=390, xaxis_title="Importance (%)", yaxis_title="", coloraxis_showscale=False, margin=dict(t=55, b=35))
        st.plotly_chart(shap_fig, use_container_width=True)

        top_three = shap_df.head(3)["Importance_Percent"].sum()
        kpi("Top three global SHAP contributions", f"{top_three:.2f}%", "Combined mean absolute SHAP importance", "violet")
        with st.expander("View top 12 SHAP feature records"):
            st.dataframe(shap_df.head(12), hide_index=True, use_container_width=True)

        st.markdown(
            """
            The global SHAP analysis used TreeExplainer on 1,000 validation
            observations. Mean absolute SHAP values measure attribution magnitude,
            not direction or causality. The leading features describe this model
            and sample; they are not a universal ranking of physical groundwater drivers.
            """
        )
    else:
        st.warning("SHAP importance is missing expected Feature, Mean_Absolute_SHAP or Importance_Percent columns.")
else:
    st.warning("SHAP importance CSV is unavailable.")


# ---------------------------------------------------------------------
# PREDICTION UNCERTAINTY
# ---------------------------------------------------------------------
st.divider()
st.subheader("8 · Prediction uncertainty")
intervals = data["prediction intervals"]
if intervals is not None and not intervals.empty:
    st.markdown(
        """
        A preliminary conformal-style interval analysis used validation residuals
        from the tuned LightGBM model. The reported 90% interval radius was
        approximately ±4.218 m, with overall test coverage of approximately 89.96%.
        """
    )
    st.caption(
        "A compact sample is shown first for readability. Open the full results "
        "only when you need to inspect all saved station and model-input fields."
    )
    preferred_columns = [
        "Station", "District", "Target_Gap_Days",
        "GWL_m_bgl", "Previous_GWL_m_bgl",
        "Target_Date", "Predicted_GWL_m_bgl",
        "Lower_Bound", "Upper_Bound",
    ]
    visible_columns = [column for column in preferred_columns if column in intervals.columns]
    if not visible_columns:
        visible_columns = list(intervals.columns[:6])
    st.dataframe(
        intervals[visible_columns].head(12),
        hide_index=True,
        use_container_width=True,
    )
    with st.expander("View the full prediction-interval results"):
        st.dataframe(intervals, hide_index=True, use_container_width=True)
    st.html(
        """
        <div class="research-caveat">
          <strong>Exploratory uncertainty only.</strong> This is not a final calibrated
          guarantee. The validation set was also used during model selection, and
          coverage varied by forecast gap. A separate calibration set and
          gap-conditional calibration are still needed for a stronger uncertainty claim.
        </div>
        """
    )
else:
    st.warning("Prediction interval results are unavailable.")


# ---------------------------------------------------------------------
# LIMITATIONS AND NEXT STEPS
# ---------------------------------------------------------------------
st.divider()
st.subheader("9 · Scientific limitations and next research steps")

left, right = st.columns(2)
with left:
    st.markdown(
        """
        **Current limitations**
        - No direct rainfall, pumping-volume, crop, recharge or aquifer-property
          predictors are included in the current 12-feature model.
        - Provisional station IDs are not confirmed official well identifiers.
        - Coordinate gaps limit mapping and geographic validation coverage.
        - Unusual readings and station-identity conflicts require further source verification.
        - The model predicts groundwater level in m bgl; it does not estimate
          groundwater availability, extraction stage or official stress category.
        - Temporal and geographic results are exploratory and require a locked
          final evaluation protocol.
        """
    )
with right:
    st.markdown(
        """
        **Recommended next steps**
        1. Verify station identity and measurement metadata against official CGWB records.
        2. Reconcile raw-source count and anomaly audit discrepancies.
        3. Establish a final untouched temporal test protocol.
        4. Evaluate external-state transfer only after acquiring compatible data.
        5. Build separate calibration data for prediction intervals.
        6. Define groundwater-stress outcomes using official assessment criteria
           and appropriate hydrogeological context.
        7. Investigate district-specific errors and unusually large prediction residuals.
        8. Add hydrogeological and climate drivers only when reliable, compatible
           data are obtained and leakage-safe feature construction is demonstrated.
        """
    )

st.html(
    """
    <div class="research-positive">
      <strong>Current research status:</strong> completed exploratory Tamil Nadu
      forecasting and district-holdout experiments. Cross-state transfer, official
      stress classification and final independent calibration remain future work.
    </div>
    """
)
