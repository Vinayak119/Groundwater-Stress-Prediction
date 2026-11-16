from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.ui import page_header, setup_page


ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT / "results" / "metrics"

FILES = {
    "benchmark": METRICS_DIR / "initial_model_benchmark.csv",
    "geographic": METRICS_DIR / "geographic_holdout_results.csv",
    "district": METRICS_DIR / "geographic_district_performance.csv",
    "gap": METRICS_DIR / "test_performance_by_gap.csv",
}


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


setup_page("Model Performance", "Evaluation")
page_header(
    "Model Performance",
    "See how the models perform on later observations, how results change across "
    "districts, and whether machine learning improves on a simple baseline.",
)

missing = [name for name, path in FILES.items() if not path.exists()]
if missing:
    st.error("Required evaluation files are missing: " + ", ".join(missing))
    st.stop()

try:
    benchmark = load_csv(str(FILES["benchmark"]))
    geographic = load_csv(str(FILES["geographic"]))
    district = load_csv(str(FILES["district"]))
    gap = load_csv(str(FILES["gap"]))
except Exception as exc:
    st.error(f"Could not load evaluation results: {exc}")
    st.stop()

# Normalise labels so filtering is consistent.
benchmark["Split"] = benchmark["Split"].astype(str).str.lower()
benchmark["Model"] = benchmark["Model"].astype(str)
geographic["Model"] = geographic["Model"].astype(str)

st.info(
    "The temporal test set and geographic district holdout answer different "
    "questions. The geographic holdout is an internal Tamil Nadu evaluation, "
    "not independent external validation."
)

st.subheader("1 · Performance on later observations")
st.caption(
    "The temporal benchmark trains on earlier data and evaluates on later periods. "
    "MAE and RMSE are measured in metres; lower values indicate smaller errors. "
    "For R², higher is generally better."
)

test_rows = benchmark[benchmark["Split"] == "test"].copy()
validation_rows = benchmark[benchmark["Split"] == "validation"].copy()

if test_rows.empty or validation_rows.empty:
    st.error("The benchmark file does not contain both validation and test rows.")
    st.stop()

test_best = test_rows.loc[test_rows["MAE"].idxmin()]
validation_best = validation_rows.loc[validation_rows["MAE"].idxmin()]

a, b, c, d = st.columns(4)
a.metric(
    "Best validation MAE",
    f'{validation_best["MAE"]:.3f} m',
    str(validation_best["Model"]),
)
b.metric(
    "Best temporal test MAE",
    f'{test_best["MAE"]:.3f} m',
    str(test_best["Model"]),
)

persistence_rows = test_rows[test_rows["Model"] == "Persistence Baseline"]
lightgbm_rows = test_rows[test_rows["Model"] == "LightGBM"]
if persistence_rows.empty or lightgbm_rows.empty:
    st.error("The benchmark is missing Persistence Baseline or LightGBM test results.")
    st.stop()

persistence_test = persistence_rows.iloc[0]
lightgbm_test = lightgbm_rows.iloc[0]
c.metric("Persistence test MAE", f'{persistence_test["MAE"]:.3f} m')
d.metric("LightGBM test MAE", f'{lightgbm_test["MAE"]:.3f} m')

st.markdown(
    "**Main takeaway:** LightGBM had the lowest validation MAE, but persistence "
    "had the lowest MAE on the later temporal test set. The initial machine-learning "
    "models therefore did not outperform the simple persistence baseline overall."
)

with st.expander("View detailed temporal benchmark results"):
    view = benchmark.copy()
    for column in ["MAE", "RMSE", "R2"]:
        view[column] = view[column].map(lambda value: f"{value:.4f}")
    st.dataframe(
        view.sort_values(["Split", "MAE"]),
        hide_index=True,
        use_container_width=True,
    )

models = [
    "Persistence Baseline",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
    "LightGBM",
]
chart_data = benchmark.copy()
chart_data["Model"] = pd.Categorical(
    chart_data["Model"], categories=models, ordered=True
)

fig_mae = px.bar(
    chart_data,
    x="Model",
    y="MAE",
    color="Split",
    barmode="group",
    title="Mean Absolute Error by model and temporal split",
    labels={"MAE": "MAE (m)", "Model": "Model"},
)
fig_mae.update_layout(
    template="plotly_dark",
    legend_title_text="Evaluation split",
    height=390,
    margin=dict(t=55, b=15),
)
st.plotly_chart(
    fig_mae, use_container_width=True, key="temporal_mae_comparison"
)
st.caption("MAE is the average absolute difference between predicted and observed levels.")

fig_rmse = px.bar(
    chart_data,
    x="Model",
    y="RMSE",
    color="Split",
    barmode="group",
    title="Root Mean Squared Error by model and temporal split",
    labels={"RMSE": "RMSE (m)", "Model": "Model"},
)
fig_rmse.update_layout(
    template="plotly_dark", height=390, margin=dict(t=55, b=15)
)
st.plotly_chart(
    fig_rmse, use_container_width=True, key="temporal_rmse_comparison"
)
st.caption(
    "RMSE penalises large errors more strongly than MAE, so it helps reveal "
    "models affected by occasional large misses."
)

fig_r2 = px.bar(
    chart_data,
    x="Model",
    y="R2",
    color="Split",
    barmode="group",
    title="R² by model and temporal split",
    labels={"R2": "R²", "Model": "Model"},
)
fig_r2.update_layout(
    template="plotly_dark", height=390, margin=dict(t=55, b=15)
)
st.plotly_chart(
    fig_r2, use_container_width=True, key="temporal_r2_comparison"
)
st.caption(
    "R² describes variation explained relative to a mean-value reference. "
    "Compare it alongside MAE and RMSE rather than using it alone."
)

st.divider()
st.subheader("2 · Testing in districts held out from training")
st.caption(
    "This test asks whether the model can generalise to selected districts it did "
    "not see during training. It is still an internal Tamil Nadu evaluation."
)

geo_p = geographic[geographic["Model"].str.lower() == "persistence"]
geo_l = geographic[geographic["Model"].str.lower() == "lightgbm"]

if not geo_p.empty and not geo_l.empty:
    p = geo_p.iloc[0]
    l = geo_l.iloc[0]
    mae_improvement = (p["MAE"] - l["MAE"]) / p["MAE"] * 100
    rmse_improvement = (p["RMSE"] - l["RMSE"]) / p["RMSE"] * 100

    x, y, z = st.columns(3)
    x.metric("Persistence MAE", f'{p["MAE"]:.3f} m')
    y.metric("LightGBM MAE", f'{l["MAE"]:.3f} m')
    z.metric("LightGBM MAE improvement", f"{mae_improvement:.1f}%")

    st.write(
        f"On this district-based holdout, LightGBM's RMSE was {l['RMSE']:.3f} m "
        f"versus {p['RMSE']:.3f} m for persistence "
        f"({rmse_improvement:.1f}% lower)."
    )
else:
    st.warning(
        "The geographic results do not contain both Persistence and LightGBM rows."
    )

geo_view = geographic.copy()
for column in ["MAE", "RMSE", "R2"]:
    if column in geo_view.columns:
        geo_view[column] = geo_view[column].map(lambda value: f"{value:.4f}")
st.dataframe(geo_view, hide_index=True, use_container_width=True)

fig_geo = px.bar(
    geographic,
    x="Model",
    y="MAE",
    title="Geographic holdout: persistence vs LightGBM",
    labels={"MAE": "MAE (m)"},
    text_auto=".3f",
)
fig_geo.update_layout(
    template="plotly_dark", height=390, margin=dict(t=55, b=15)
)
st.plotly_chart(
    fig_geo, use_container_width=True, key="geographic_holdout_model_chart"
)

st.caption(
    "The geographic holdout used seven held-out Tamil Nadu districts. "
    "It is not a cross-state or independent external validation."
)

st.subheader("Which districts were harder to predict?")
st.caption(
    "District-level errors help identify where model predictions were less accurate. "
    "They do not explain the cause of those errors."
)

district = district.sort_values("MAE", ascending=True)
district_display = district.copy()
for column in ["MAE", "RMSE"]:
    if column in district_display.columns:
        district_display[column] = district_display[column].map(
            lambda value: f"{value:.3f}"
        )
st.dataframe(district_display, hide_index=True, use_container_width=True)

fig_district = px.bar(
    district.sort_values("RMSE", ascending=True),
    x="RMSE",
    y="District",
    orientation="h",
    title="LightGBM RMSE by held-out district",
    labels={"RMSE": "RMSE (m)"},
    hover_data=["Stations", "Predictions", "MAE"],
    text_auto=".2f",
)
fig_district.update_layout(
    template="plotly_dark", height=430, margin=dict(t=55, b=15)
)
st.plotly_chart(
    fig_district,
    use_container_width=True,
    key="geographic_district_rmse_chart",
)

karur = district[district["District"].astype(str).str.lower() == "karur"]
if not karur.empty:
    st.warning(
        f"Karur has the highest district RMSE in this result "
        f"({float(karur.iloc[0]['RMSE']):.3f} m). This indicates substantial "
        "district-level error variation and warrants further investigation."
    )

st.divider()
st.subheader("3 · Does the time between observations matter?")
st.write(
    "The model predicts the next available measurement, so the time gap is not "
    "always the same. These results compare persistence and LightGBM across four "
    "gap ranges."
)

gap_order = ["0-60 days", "61-90 days", "91-120 days", "121-150 days"]
gap["Gap_Group"] = pd.Categorical(
    gap["Gap_Group"].astype(str), categories=gap_order, ordered=True
)
gap_display = gap.sort_values(["Gap_Group", "Model"]).copy()
for column in ["MAE", "RMSE", "R2"]:
    if column in gap_display.columns:
        gap_display[column] = gap_display[column].map(
            lambda value: f"{value:.3f}"
        )
st.dataframe(gap_display, hide_index=True, use_container_width=True)

fig_gap = px.bar(
    gap,
    x="Gap_Group",
    y="MAE",
    color="Model",
    barmode="group",
    category_orders={"Gap_Group": gap_order},
    title="MAE by forecast gap: persistence vs LightGBM",
    labels={"Gap_Group": "Time to next observation", "MAE": "MAE (m)"},
    text_auto=".2f",
)
fig_gap.update_layout(
    template="plotly_dark", height=390, margin=dict(t=55, b=15)
)
st.plotly_chart(
    fig_gap, use_container_width=True, key="forecast_gap_mae_chart"
)

st.markdown(
    """
    **Interpretation:** Persistence has lower MAE in the first three gap groups.
    LightGBM has a slightly lower MAE in the 121–150 day group, but its RMSE is
    higher in that group. The result does not establish a consistent advantage
    for LightGBM at longer gaps.
    """
)

st.divider()
st.subheader("4 · What should we take away?")
st.markdown(
    """
    - **Validation is not the final test:** the lowest validation error does not
      guarantee the best performance on later observations.
    - **Persistence is a serious baseline:** a model should demonstrate improvement
      over simply carrying forward the latest measured groundwater level.
    - **Geographic results vary:** the district holdout shows an overall LightGBM
      advantage, but errors differ across districts.
    - **Outliers matter:** unusually large groundwater-level changes can strongly
      affect RMSE and require source-record and station-identity investigation.
    - **Forecast horizons vary:** the target is the next available observation
      within 150 days, not a fixed quarterly or 90-day forecast.
    - **No stress classification is evaluated here:** these are groundwater-level
      regression metrics, not official groundwater-resource assessment categories.
    - **Further validation is required:** results are exploratory and should be
      followed by a locked final evaluation and, where possible, independent data.
    """
)

st.subheader("Download evaluation results")
st.caption("Download the exact saved result tables used in the charts above.")

st.download_button(
    "Download temporal benchmark CSV",
    data=benchmark.to_csv(index=False).encode("utf-8"),
    file_name="temporal_model_benchmark.csv",
    mime="text/csv",
    key="download_temporal_benchmark_csv",
)

st.download_button(
    "Download geographic holdout CSV",
    data=geographic.to_csv(index=False).encode("utf-8"),
    file_name="geographic_holdout_results.csv",
    mime="text/csv",
    key="download_geographic_holdout_csv",
)
