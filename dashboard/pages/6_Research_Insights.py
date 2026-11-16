from pathlib import Path
import json
import pandas as pd
import streamlit as st

from dashboard.components.ui import setup_page, page_header

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


@st.cache_data
def read_csv(path):
    return pd.read_csv(path)


def count_csv(path):
    try:
        return len(read_csv(str(path)))
    except Exception:
        return None


setup_page("Research Insights", "Research Summary")
page_header(
    "Research Insights & Experimental Findings",
    "A consolidated research view of CGWB Tamil Nadu data preparation, groundwater-level forecasting, validation, explainability and limitations.",
)

st.info(
    "This page reports completed experiments from the current project artefacts. "
    "It distinguishes measured results from proposed research questions and "
    "does not present groundwater-level predictions as official stress classifications."
)

st.subheader("Research status")

status_cols = st.columns(4)
for col, (name, path) in zip(status_cols, list(FILES.items())[:4]):
    col.metric(name.title(), "Available" if path.exists() else "Not found")

missing = [name for name, path in FILES.items() if not path.exists()]
if missing:
    st.warning("Some research artefacts are not present: " + ", ".join(missing))

clean_count = count_csv(FILES["cleaned observations"])
forecast_count = count_csv(FILES["purged forecasting pairs"])
quarantine_count = count_csv(FILES["quarantined observations"])

a, b, c = st.columns(3)
a.metric("Cleaned observations", f"{clean_count:,}" if clean_count is not None else "Unavailable")
b.metric("Quarantined observations", f"{quarantine_count:,}" if quarantine_count is not None else "Unavailable")
c.metric("Temporally purged forecast pairs", f"{forecast_count:,}" if forecast_count is not None else "Unavailable")

st.divider()
st.subheader("1. Research questions and current evidence")

research_questions = [
    (
        "RQ1 — Temporal forecasting performance",
        "Do machine-learning regressors improve on a persistence baseline when predicting the next available groundwater observation?",
        "Completed initial temporal benchmark. Persistence achieved lower overall test MAE than all four initial machine-learning models."
    ),
    (
        "RQ2 — Geographic generalisation within Tamil Nadu",
        "How does LightGBM perform on districts withheld from model training?",
        "Completed seven-district geographic holdout. LightGBM outperformed persistence overall on this evaluation, but district-level errors varied."
    ),
    (
        "RQ3 — Model interpretability",
        "Which input features contribute most strongly to LightGBM predictions?",
        "Completed global SHAP analysis on 1,000 validation observations. Current groundwater level, historical mean and current month had the largest mean absolute SHAP contributions."
    ),
    (
        "RQ4 — Forecast-gap sensitivity",
        "Does model performance change according to the time until the next available observation?",
        "Completed four forecast-gap comparisons. Persistence had lower MAE in the first three groups; LightGBM had marginally lower MAE in the final group but higher RMSE."
    ),
    (
        "RQ5 — Cross-state transfer",
        "Can a model trained in Tamil Nadu generalise to a geographically unseen neighbouring state?",
        "Not yet evaluated. No Karnataka or Andhra Pradesh external-state validation result is available in the current artefacts."
    ),
]

for title, question, evidence in research_questions:
    with st.expander(title, expanded=False):
        st.markdown("**Research question**")
        st.write(question)
        st.markdown("**Current evidence**")
        st.write(evidence)

st.divider()
st.subheader("2. Data sources, cleaning and audit")

st.markdown(
    """
    The project uses CGWB manual groundwater-level observations for Tamil Nadu,
    combining historical and recent source files. The cleaned observation table
    is the basis for the forecasting feature-generation process.

    **Observed data characteristics**
    - 77,024 cleaned observations.
    - 2,318 provisional station identifiers.
    - 33 districts represented.
    - Historical observed dates begin in January 1996; the recent source extends
      through January 2025.
    - 246 records have missing coordinates. These can contribute to temporal
      modelling but cannot support coordinate-based mapping or geographic grouping.
    - RL_MSL is largely missing and is not treated as a general model predictor.
    """
)

st.markdown("**Data-quality decisions**")
st.markdown(
    """
    - Repeated provisional-station/calendar-date conflicts were quarantined rather
      than averaged.
    - Maruthur2 records were quarantined pending confirmation of official well identity.
    - Unusual high and negative groundwater-level readings were retained for review
      rather than automatically deleted.
    - Negative readings may represent real conditions and require interpretation
      using measurement context.
    - Provisional station identifiers must not be assumed to be verified official
      well identities.
    """
)

with st.expander("View available data audit artefacts"):
    for name in [
        "observations_model_ready.csv",
        "observations_quarantined.csv",
        "calendar_date_conflict_groups.csv",
    ]:
        candidate = DATA / name if name != "calendar_date_conflict_groups.csv" else REPORTS / name
        st.write(f"{name}: {'Available' if candidate.exists() else 'Not found'}")

st.divider()
st.subheader("3. Forecasting methodology and temporal split")

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

    The historical features are constructed from observations at or before the
    forecast origin. The future target date and target gap are not model inputs.
    """
)

split_rows = [
    {"Split": "Training", "Rule": "Target date through 2018-12-31", "Rows": "51,816"},
    {"Split": "Validation", "Rule": "Input date from 2019-01-01; target through 2020-12-31", "Rows": "3,227"},
    {"Split": "Test", "Rule": "Input date from 2021-01-01 onward", "Rows": "8,350"},
    {"Split": "Boundary exclusions", "Rule": "Pairs removed to prevent split overlap", "Rows": "1,041"},
]
st.dataframe(pd.DataFrame(split_rows), hide_index=True, use_container_width=True)

st.warning(
    "The test split uses forecast origins from 2021 onward and may use earlier "
    "observations from the same station as historical context. This is a prospective "
    "forecasting setup, not a fully independent station or external-state test. "
    "The test results have also been inspected during exploratory model development."
)

st.divider()
st.subheader("4. Temporal benchmark findings")

if FILES["temporal benchmark"].exists():
    benchmark = read_csv(str(FILES["temporal benchmark"]))
    test = benchmark[benchmark["Split"].astype(str).str.lower() == "test"].copy()
    validation = benchmark[benchmark["Split"].astype(str).str.lower() == "validation"].copy()

    if not test.empty and not validation.empty:
        best_val = validation.loc[validation["MAE"].idxmin()]
        best_test = test.loc[test["MAE"].idxmin()]
        p = test[test["Model"] == "Persistence Baseline"].iloc[0]
        l = test[test["Model"] == "LightGBM"].iloc[0]

        x, y, z = st.columns(3)
        x.metric("Best validation MAE", f'{best_val["MAE"]:.3f} m', str(best_val["Model"]))
        y.metric("Best test MAE", f'{best_test["MAE"]:.3f} m', str(best_test["Model"]))
        z.metric("Persistence test MAE", f'{p["MAE"]:.3f} m')

        st.write(
            f"LightGBM test MAE was {l['MAE']:.3f} m, compared with "
            f"{p['MAE']:.3f} m for persistence. Persistence therefore had "
            f"{(l['MAE'] - p['MAE']) / p['MAE'] * 100:.1f}% lower MAE "
            "than LightGBM on this temporal test set."
        )

    st.dataframe(benchmark, hide_index=True, use_container_width=True)
else:
    st.warning("Temporal benchmark CSV is unavailable.")

st.divider()
st.subheader("5. Geographic holdout findings")

if FILES["geographic holdout"].exists():
    geo = read_csv(str(FILES["geographic holdout"]))
    st.dataframe(geo, hide_index=True, use_container_width=True)

    gp = geo[geo["Model"].astype(str).str.lower() == "persistence"]
    gl = geo[geo["Model"].astype(str).str.lower() == "lightgbm"]

    if not gp.empty and not gl.empty:
        p = gp.iloc[0]
        l = gl.iloc[0]
        st.write(
            f"On the district holdout, LightGBM MAE was {l['MAE']:.3f} m versus "
            f"{p['MAE']:.3f} m for persistence. LightGBM RMSE was "
            f"{l['RMSE']:.3f} m versus {p['RMSE']:.3f} m."
        )

if FILES["district results"].exists():
    st.markdown("**District-level LightGBM results**")
    district = read_csv(str(FILES["district results"])).sort_values("RMSE", ascending=False)
    st.dataframe(district, hide_index=True, use_container_width=True)
    if not district.empty:
        worst = district.iloc[0]
        st.warning(
            f"{worst['District']} has the highest district RMSE in this evaluation "
            f"({worst['RMSE']:.3f} m). This warrants further error and source-record review."
        )

st.caption(
    "This is an internal Tamil Nadu district holdout, not a cross-state transfer test "
    "or independent external validation."
)

st.divider()
st.subheader("6. Explainability findings")

if FILES["SHAP importance"].exists():
    shap_df = read_csv(str(FILES["SHAP importance"]))
    shap_df = shap_df.sort_values("Mean_Absolute_SHAP", ascending=False)
    st.dataframe(shap_df.head(12), hide_index=True, use_container_width=True)

    if len(shap_df) >= 3:
        top_three = shap_df.head(3)["Importance_Percent"].sum()
        st.metric("Top three global SHAP contributions", f"{top_three:.2f}%")

    st.markdown(
        """
        The global SHAP analysis used TreeExplainer on 1,000 validation observations.
        Mean absolute SHAP values measure attribution magnitude, not direction or
        causality. The leading features describe this model and sample; they are not
        a universal ranking of physical groundwater drivers.
        """
    )
else:
    st.warning("SHAP importance CSV is unavailable.")

st.divider()
st.subheader("7. Prediction uncertainty")

if FILES["prediction intervals"].exists():
    intervals = read_csv(str(FILES["prediction intervals"]))
    st.markdown(
        """
        A preliminary conformal-style interval analysis used validation residuals
        from the tuned LightGBM model. The reported 90% interval radius was
        approximately ±4.218 m, with overall test coverage of approximately 89.96%.
        """
    )
    st.dataframe(intervals, hide_index=True, use_container_width=True)
    st.warning(
        "This is exploratory uncertainty analysis, not a final calibrated guarantee. "
        "The validation set was also used during model selection, and coverage varied "
        "by forecast gap. A separate calibration set and gap-conditional calibration "
        "are still needed for a stronger uncertainty claim."
    )
else:
    st.warning("Prediction interval results are unavailable.")

st.divider()
st.subheader("8. Scientific limitations and next research steps")

st.markdown(
    """
    **Current limitations**
    - No direct rainfall, pumping-volume, crop, recharge or aquifer-property predictors
      are included in the current 12-feature model.
    - Provisional station IDs are not confirmed official well identifiers.
    - Coordinate gaps limit mapping and geographic validation coverage.
    - Unusual readings and station-identity conflicts require further source verification.
    - The model predicts groundwater level in m bgl; it does not estimate groundwater
      availability, extraction stage or official stress category.
    - Temporal and geographic results are exploratory and require a locked final
      evaluation protocol.

    **Recommended next steps**
    1. Verify station identity and measurement metadata against official CGWB records.
    2. Reconcile raw-source count and anomaly audit discrepancies.
    3. Establish a final untouched temporal test protocol.
    4. Evaluate external-state transfer only after acquiring compatible data.
    5. Build separate calibration data for prediction intervals.
    6. Define groundwater-stress outcomes using official assessment criteria and
       appropriate hydrogeological context.
    7. Investigate district-specific errors and unusually large prediction residuals.
    8. Add hydrogeological and climate drivers only when reliable, compatible data
       are obtained and leakage-safe feature construction is demonstrated.
    """
)

st.caption(
    "Research status: completed exploratory Tamil Nadu forecasting and district "
    "holdout experiments. Cross-state transfer, official stress classification and "
    "final independent calibration remain future work."
)




