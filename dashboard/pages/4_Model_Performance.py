from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from dashboard.components.ui import setup_page, page_header

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


def metric_value(df, model, split, metric):
    row = df[(df["Model"] == model) & (df["Split"] == split)]
    return float(row.iloc[0][metric]) if not row.empty else np.nan


setup_page("Model Performance", "Evaluation")
page_header(
    "Model Performance & Evaluation",
    "Evidence-based comparison of temporal forecasting models, geographic generalisation and forecast-gap behaviour.",
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

# Normalise labels for reliable filtering.
benchmark["Split"] = benchmark["Split"].astype(str).str.lower()
benchmark["Model"] = benchmark["Model"].astype(str)
geographic["Model"] = geographic["Model"].astype(str)

st.info(
    "Evaluation note: the temporal test set and geographic district holdout answer "
    "different questions. The geographic holdout is an internal district-based "
    "evaluation, not independent external validation."
)

st.subheader("1. Temporal benchmark")

models = ["Persistence Baseline", "Decision Tree", "Random Forest", "XGBoost", "LightGBM"]
test_rows = benchmark[benchmark["Split"] == "test"].copy()
validation_rows = benchmark[benchmark["Split"] == "validation"].copy()

if test_rows.empty or validation_rows.empty:
    st.error("The benchmark file does not contain both validation and test rows.")
    st.stop()

test_best = test_rows.loc[test_rows["MAE"].idxmin()]
validation_best = validation_rows.loc[validation_rows["MAE"].idxmin()]

a, b, c, d = st.columns(4)
a.metric("Best validation MAE", f'{validation_best["MAE"]:.3f} m', str(validation_best["Model"]))
b.metric("Best temporal test MAE", f'{test_best["MAE"]:.3f} m', str(test_best["Model"]))
persistence_test = test_rows[test_rows["Model"] == "Persistence Baseline"].iloc[0]
lightgbm_test = test_rows[test_rows["Model"] == "LightGBM"].iloc[0]
c.metric("Persistence test MAE", f'{persistence_test["MAE"]:.3f} m')
d.metric("LightGBM test MAE", f'{lightgbm_test["MAE"]:.3f} m')

st.markdown(
    "**Main finding:** LightGBM achieved the lowest validation MAE, but the persistence "
    "baseline achieved the lowest test MAE. The initial ML models therefore did not "
    "beat persistence on the overall temporal test set."
)

with st.expander("View complete temporal benchmark table", expanded=True):
    view = benchmark.copy()
    view["MAE"] = view["MAE"].map(lambda x: f"{x:.4f}")
    view["RMSE"] = view["RMSE"].map(lambda x: f"{x:.4f}")
    view["R2"] = view["R2"].map(lambda x: f"{x:.4f}")
    st.dataframe(
        view.sort_values(["Split", "MAE"]),
        hide_index=True,
        use_container_width=True,
    )

chart_data = benchmark.copy()
chart_data["Model"] = pd.Categorical(
    chart_data["Model"],
    categories=models,
    ordered=True,
)
fig = px.bar(
    chart_data,
    x="Model",
    y="MAE",
    color="Split",
    barmode="group",
    title="Mean Absolute Error by model and temporal split",
    labels={"MAE": "MAE (m)", "Model": "Model"},
)
fig.update_layout(legend_title_text="Evaluation split")
st.plotly_chart(fig, use_container_width=True)

fig_rmse = px.bar(
    chart_data,
    x="Model",
    y="RMSE",
    color="Split",
    barmode="group",
    title="Root Mean Squared Error by model and temporal split",
    labels={"RMSE": "RMSE (m)", "Model": "Model"},
)
st.plotly_chart(fig_rmse, use_container_width=True)

fig_r2 = px.bar(
    chart_data,
    x="Model",
    y="R2",
    color="Split",
    barmode="group",
    title="R² by model and temporal split",
    labels={"R2": "R²", "Model": "Model"},
)
st.plotly_chart(fig_r2, use_container_width=True)

st.divider()
st.subheader("2. Geographic district holdout")

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

geo_view = geographic.copy()
for col in ["MAE", "RMSE", "R2"]:
    geo_view[col] = geo_view[col].map(lambda x: f"{x:.4f}")
st.dataframe(geo_view, hide_index=True, use_container_width=True)

fig_geo = px.bar(
    geographic,
    x="Model",
    y="MAE",
    title="Geographic holdout: persistence vs LightGBM",
    labels={"MAE": "MAE (m)"},
    text_auto=".3f",
)
st.plotly_chart(fig_geo, use_container_width=True)

st.caption(
    "The geographic holdout used seven held-out Tamil Nadu districts. "
    "It is not a cross-state or independent external validation."
)

st.subheader("District-level LightGBM performance")

district = district.sort_values("MAE", ascending=True)
district_display = district.copy()
for col in ["MAE", "RMSE"]:
    district_display[col] = district_display[col].map(lambda x: f"{x:.3f}")

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
st.plotly_chart(fig_district, use_container_width=True)

karur = district[district["District"].astype(str).str.lower() == "karur"]
if not karur.empty:
    st.warning(
        f"Karur has the highest district RMSE in this result "
        f"({float(karur.iloc[0]['RMSE']):.3f} m). This indicates substantial "
        "district-level error variation and warrants further investigation."
    )

st.divider()
st.subheader("3. Performance by forecast gap")

st.write(
    "The next-observation target can occur at different intervals. "
    "The table and chart below compare persistence and LightGBM across the "
    "four forecast-gap groups."
)

gap_order = ["0-60 days", "61-90 days", "91-120 days", "121-150 days"]
gap["Gap_Group"] = pd.Categorical(
    gap["Gap_Group"].astype(str),
    categories=gap_order,
    ordered=True,
)
gap_display = gap.sort_values(["Gap_Group", "Model"]).copy()
for col in ["MAE", "RMSE", "R2"]:
    gap_display[col] = gap_display[col].map(lambda x: f"{x:.3f}")
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
st.plotly_chart(fig_gap, use_container_width=True)

st.markdown(
    """
    **Interpretation:** Persistence has lower MAE in the first three gap groups.
    LightGBM has a slightly lower MAE in the 121–150 day group, but its RMSE is
    higher in that group. The result does not establish a consistent advantage
    for LightGBM at longer gaps.
    """
)

st.divider()
st.subheader("4. Research interpretation and limitations")

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

st.download_button(
    "Download temporal benchmark CSV",
    data=benchmark.to_csv(index=False).encode("utf-8"),
    file_name="temporal_model_benchmark.csv",
    mime="text/csv",
)

st.download_button(
    "Download geographic holdout CSV",
    data=geographic.to_csv(index=False).encode("utf-8"),
    file_name="geographic_holdout_results.csv",
    mime="text/csv",
)
