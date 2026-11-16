from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.ui import setup_page, page_header

ROOT = Path(__file__).resolve().parents[2]
SHAP_DIR = ROOT / "results" / "shap"

IMPORTANCE_PATH = SHAP_DIR / "shap_feature_importance.csv"
SUMMARY_PATH = SHAP_DIR / "shap_summary.png"
BAR_PATH = SHAP_DIR / "shap_bar.png"


@st.cache_data
def load_importance(path):
    df = pd.read_csv(path)
    df["Mean_Absolute_SHAP"] = pd.to_numeric(
        df["Mean_Absolute_SHAP"], errors="coerce"
    )
    df["Importance_Percent"] = pd.to_numeric(
        df["Importance_Percent"], errors="coerce"
    )
    return df.dropna(subset=["Mean_Absolute_SHAP"]).sort_values(
        "Mean_Absolute_SHAP", ascending=False
    )


setup_page("Explainable AI", "Model Interpretability")
page_header(
    "Explainable AI – SHAP Analysis",
    "Understand which input features contribute most to the LightGBM groundwater-level model's predictions.",
)

if not IMPORTANCE_PATH.exists():
    st.error(f"SHAP importance file not found: {IMPORTANCE_PATH}")
    st.stop()

try:
    importance = load_importance(str(IMPORTANCE_PATH))
except Exception as exc:
    st.error(f"Could not load SHAP results: {exc}")
    st.stop()

if importance.empty:
    st.warning("The SHAP importance file contains no usable feature values.")
    st.stop()

st.info(
    "These results are global SHAP feature attributions calculated from a sample "
    "of 1,000 validation observations using TreeExplainer. They describe model "
    "behaviour, not physical causation or official groundwater-stress categories."
)

st.subheader("1. Global feature importance")

top = importance.iloc[0]
top_three_share = importance.head(3)["Importance_Percent"].sum()

a, b, c = st.columns(3)
a.metric("Most influential feature", str(top["Feature"]))
b.metric("Top feature importance", f'{top["Importance_Percent"]:.2f}%')
c.metric("Top three combined", f"{top_three_share:.2f}%")

fig = px.bar(
    importance.head(12).sort_values("Mean_Absolute_SHAP"),
    x="Mean_Absolute_SHAP",
    y="Feature",
    orientation="h",
    title="Top 12 features by mean absolute SHAP value",
    labels={
        "Mean_Absolute_SHAP": "Mean absolute SHAP value",
        "Feature": "Model input feature",
    },
    hover_data={"Importance_Percent": ":.3f"},
    text_auto=".3f",
)
fig.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Larger mean absolute SHAP values indicate greater average influence on "
    "the model output across the analysed validation sample. Absolute values "
    "do not show the direction of influence."
)

st.subheader("2. Saved SHAP visualisations")

if SUMMARY_PATH.exists():
    st.markdown("**SHAP summary plot**")
    st.image(
        str(SUMMARY_PATH),
        caption="Saved SHAP summary plot from the validation analysis.",
        use_container_width=True,
    )
else:
    st.warning("The saved SHAP summary image is unavailable.")

if BAR_PATH.exists():
    st.markdown("**SHAP feature-importance plot**")
    st.image(
        str(BAR_PATH),
        caption="Saved SHAP bar plot from the validation analysis.",
        use_container_width=True,
    )
else:
    st.warning("The saved SHAP bar image is unavailable.")

st.divider()
st.subheader("3. Feature attribution table")

display = importance.copy()
display["Mean_Absolute_SHAP"] = display["Mean_Absolute_SHAP"].map(
    lambda value: f"{value:.6f}"
)
display["Importance_Percent"] = display["Importance_Percent"].map(
    lambda value: f"{value:.4f}%"
)
st.dataframe(display, hide_index=True, use_container_width=True)

st.download_button(
    "Download SHAP feature-importance CSV",
    data=importance.to_csv(index=False).encode("utf-8"),
    file_name="shap_feature_importance.csv",
    mime="text/csv",
)

st.divider()
st.subheader("4. Research interpretation")

st.markdown(
    """
    **Main observations from this analysis**

    - Current groundwater level (`GWL_m_bgl`) has the largest mean absolute SHAP
      contribution in the analysed sample.
    - Historical mean groundwater level is the second-largest contributor.
    - `Current_Month` has a substantial contribution, indicating that the model
      uses the month feature when generating its predictions.
    - Previous and two-observations-back groundwater levels also contribute to
      model output.
    - Latitude and longitude have smaller global attribution values in this
      particular analysis.
    - Missing-value indicator features have very small global contributions
      compared with the main numerical features.

    **Important interpretation limits**

    - Mean absolute SHAP values measure attribution magnitude, not direction.
    - These values do not establish that a feature causes groundwater-level change.
    - Feature importance can be affected by correlated inputs and the training data.
    - The analysis describes this trained LightGBM model and the analysed validation
      sample; it is not a universal ranking of hydrogeological drivers.
    - SHAP explanations do not convert a predicted water level into an official
      groundwater-stress assessment.
    """
)

st.warning(
    "This page presents global explanations only. A local, station-specific "
    "explanation requires the corresponding observation-level SHAP values, "
    "which are not among the currently available SHAP artefacts."
)
