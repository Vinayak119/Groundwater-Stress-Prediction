from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.ui import setup_page, page_header
from dashboard.components.styles import PLOTLY_TEMPLATE

ROOT = Path(__file__).resolve().parents[2]
SHAP_DIR = ROOT / "results" / "shap"

IMPORTANCE_PATH = SHAP_DIR / "shap_feature_importance.csv"
SUMMARY_PATH = SHAP_DIR / "shap_summary.png"
BAR_PATH = SHAP_DIR / "shap_bar.png"

# Plain-language descriptions for the model inputs used in this project.
FEATURE_DESCRIPTIONS = {
    "GWL_m_bgl": "Most recent groundwater depth, in metres below ground level.",
    "Previous_GWL_m_bgl": "Groundwater depth from the preceding available observation.",
    "Two_Observations_Back_GWL_m_bgl": "Groundwater depth from two observations earlier.",
    "Historical_Mean_GWL_m_bgl": "Mean groundwater depth across the available station history.",
    "Historical_Std_GWL_m_bgl": "Historical variability in groundwater depth at the station.",
    "History_Count": "Number of historical observations available to the model.",
    "Days_Since_Previous_Observation": "Elapsed days since the preceding observation.",
    "Current_Month": "Calendar month associated with the current observation.",
    "Month_Sin": "Cyclical sine encoding of the calendar month.",
    "Month_Cos": "Cyclical cosine encoding of the calendar month.",
    "Latitude_num": "Station latitude supplied as a geographic input.",
    "Longitude_num": "Station longitude supplied as a geographic input.",
}


@st.cache_data(show_spinner=False)
def load_importance(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"Feature", "Mean_Absolute_SHAP", "Importance_Percent"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Missing required column(s): " + ", ".join(sorted(missing))
        )

    df["Feature"] = df["Feature"].astype(str)
    df["Mean_Absolute_SHAP"] = pd.to_numeric(
        df["Mean_Absolute_SHAP"], errors="coerce"
    )
    df["Importance_Percent"] = pd.to_numeric(
        df["Importance_Percent"], errors="coerce"
    )

    return (
        df.dropna(subset=["Mean_Absolute_SHAP", "Importance_Percent"])
        .sort_values("Mean_Absolute_SHAP", ascending=False)
        .reset_index(drop=True)
    )


def pretty_feature(feature: str) -> str:
    return str(feature).replace("_", " ")


def feature_description(feature: str) -> str:
    return FEATURE_DESCRIPTIONS.get(
        str(feature),
        "Model input included in the global SHAP analysis.",
    )


def render_metric(label: str, value: str, detail: str, accent: str) -> None:
    st.html(
        f"""
        <div class="shap-kpi shap-kpi-{accent}">
            <div class="shap-kpi-label">{label}</div>
            <div class="shap-kpi-value">{value}</div>
            <div class="shap-kpi-detail">{detail}</div>
        </div>
        """
    )


setup_page("Explainable AI", "Model Interpretability")
page_header(
    "Why Does the Model Make Its Predictions?",
    "Explore which information the LightGBM model relies on most when estimating groundwater levels."
)

st.html(
    """
    <style>
    .main .block-container {max-width: 1450px; padding-top: 1.4rem;}
    .shap-intro {
        background: linear-gradient(120deg, rgba(167,139,250,.15), rgba(52,211,153,.06));
        border: 1px solid rgba(167,139,250,.28);
        border-radius: 15px;
        padding: 1.1rem 1.25rem;
        margin: .5rem 0 1.2rem;
        color: #DAD7E8;
        line-height: 1.65;
        font-size: .9rem;
    }
    .shap-kpi {
        background: linear-gradient(145deg,#242039,#181824);
        border: 1px solid #39304F;
        border-radius: 14px;
        padding: 1.05rem 1.15rem;
        min-height: 126px;
        margin-bottom: .7rem;
    }
    .shap-kpi-label {
        color: #B4B4C2;
        font-size: .70rem;
        font-weight: 750;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    .shap-kpi-value {
        color: #F8FAFC;
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -.04em;
        margin-top: .45rem;
        overflow-wrap: anywhere;
    }
    .shap-kpi-detail {
        color: #A1A1AA;
        font-size: .76rem;
        line-height: 1.45;
        margin-top: .25rem;
    }
    .shap-section-note {
        color: #A1A1AA;
        font-size: .82rem;
        line-height: 1.6;
        margin: -.35rem 0 .9rem;
    }
    .shap-feature {
        background: rgba(36,32,57,.72);
        border: 1px solid #39304F;
        border-radius: 11px;
        padding: .8rem .95rem;
        margin: .35rem 0;
    }
    .shap-feature-name {color:#F8FAFC;font-weight:700;font-size:.88rem;}
    .shap-feature-desc {color:#A1A1AA;font-size:.77rem;line-height:1.5;margin-top:.25rem;}
    .shap-caveat {
        background: rgba(120,78,28,.18);
        border: 1px solid rgba(245,190,101,.35);
        border-left: 4px solid #F4BE65;
        border-radius: 10px;
        padding: .95rem 1.1rem;
        color: #FDE7B0;
        font-size: .83rem;
        line-height: 1.65;
        margin: .7rem 0;
    }
    </style>
    """
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

st.html(
    """
    <div class="shap-intro">
        <strong>How to read this page.</strong>
        SHAP helps explain which model inputs were most influential across a
        sample of 1,000 validation observations. A higher importance value means
        the model relied more heavily on that input in this analysis. It does not
        show cause and effect, and it does not define official groundwater-stress
        categories.
    </div>
    """
)

# ---------------------------------------------------------------------
# GLOBAL IMPORTANCE
# ---------------------------------------------------------------------
st.subheader("1 · Which inputs matter most to the model?")
st.caption(
    "Mean absolute SHAP measures average attribution magnitude across the "
    "analysed sample. It does not indicate whether a feature raises or lowers "
    "a prediction."
)

top = importance.iloc[0]
top_three_share = importance.head(3)["Importance_Percent"].sum()
feature_count = len(importance)

k1, k2, k3 = st.columns(3)
with k1:
    render_metric(
        "Leading feature",
        pretty_feature(top["Feature"]),
        f'{top["Importance_Percent"]:.2f}% of total importance',
        "violet",
    )
with k2:
    render_metric(
        "Top three combined",
        f"{top_three_share:.2f}%",
        "Share of reported global importance",
        "green",
    )
with k3:
    render_metric(
        "Features analysed",
        f"{feature_count}",
        "Features with usable SHAP importance values",
        "blue",
    )

chart_data = importance.head(12).copy()
chart_data["Feature label"] = chart_data["Feature"].map(pretty_feature)
chart_data = chart_data.sort_values("Mean_Absolute_SHAP", ascending=True)

fig = px.bar(
    chart_data,
    x="Mean_Absolute_SHAP",
    y="Feature label",
    orientation="h",
    color="Importance_Percent",
    color_continuous_scale=["#4C3B73", "#A78BFA", "#34D399"],
    template=PLOTLY_TEMPLATE,
    text="Importance_Percent",
    hover_data={"Mean_Absolute_SHAP": ":.5f", "Importance_Percent": ":.2f"},
)
fig.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside",
    cliponaxis=False,
    hovertemplate=(
        "%{y}<br>Mean |SHAP|: %{x:.5f}"
        "<br>Importance: %{marker.color:.2f}%<extra></extra>"
    ),
)
fig.update_layout(
    height=max(390, 34 * len(chart_data) + 100),
    coloraxis_showscale=False,
    xaxis_title="Mean absolute SHAP value",
    yaxis_title="",
    margin=dict(t=18, b=45, l=12, r=65),
)
st.plotly_chart(fig, use_container_width=True)

top_summary = importance.head(3)[
    ["Feature", "Importance_Percent", "Mean_Absolute_SHAP"]
].copy()
top_summary["Feature"] = top_summary["Feature"].map(pretty_feature)
top_summary = top_summary.rename(columns={
    "Feature": "Input feature",
    "Importance_Percent": "Importance share (%)",
    "Mean_Absolute_SHAP": "Mean absolute SHAP",
})
st.markdown("**Three most influential inputs**")
st.dataframe(
    top_summary.round({
        "Importance share (%)": 2,
        "Mean absolute SHAP": 5,
    }),
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "The percentage labels show each feature's share of the reported global "
    "importance. Longer bars mean a feature had a larger average SHAP contribution "
    "magnitude in the analysed validation sample."
)

# ---------------------------------------------------------------------
# FEATURE GUIDE
# ---------------------------------------------------------------------
st.divider()
st.subheader("2 · Understand the important inputs")
st.markdown(
    '<div class="shap-section-note">Plain-language descriptions of the leading '
    'features. These are descriptions of model inputs, not causal claims.</div>',
    unsafe_allow_html=True,
)

for _, row in importance.head(6).iterrows():
    feature = str(row["Feature"])
    with st.expander(
        f"{pretty_feature(feature)} · {row['Importance_Percent']:.2f}% importance"
    ):
        st.write(feature_description(feature))

# ---------------------------------------------------------------------
# SAVED VISUALISATIONS
# ---------------------------------------------------------------------
st.divider()
st.subheader("3 · SHAP analysis plots")
st.caption(
    "These are the saved plots produced by the project's SHAP analysis. "
    "If a plot file is absent, the page reports that rather than generating "
    "a substitute."
)

image_left, image_right = st.columns(2, gap="large")

with image_left:
    st.markdown("**SHAP summary plot**")
    if SUMMARY_PATH.exists():
        st.image(
            str(SUMMARY_PATH),
            caption="Saved SHAP summary plot from the validation analysis.",
            use_container_width=True,
        )
    else:
        st.info("The saved SHAP summary image is unavailable.")

with image_right:
    st.markdown("**SHAP feature-importance plot**")
    if BAR_PATH.exists():
        st.image(
            str(BAR_PATH),
            caption="Saved SHAP bar plot from the validation analysis.",
            use_container_width=True,
        )
    else:
        st.info("The saved SHAP bar image is unavailable.")

# ---------------------------------------------------------------------
# TABLE AND DOWNLOAD
# ---------------------------------------------------------------------
st.divider()
st.subheader("4 · All feature-importance results")

display = importance.copy()
display["Feature"] = display["Feature"].map(pretty_feature)
display["Mean_Absolute_SHAP"] = display["Mean_Absolute_SHAP"].map(
    lambda value: f"{value:.6f}"
)
display["Importance_Percent"] = display["Importance_Percent"].map(
    lambda value: f"{value:.4f}%"
)

st.dataframe(
    display,
    hide_index=True,
    use_container_width=True,
)

st.download_button(
    "Download SHAP feature-importance CSV",
    data=importance.to_csv(index=False).encode("utf-8"),
    file_name="shap_feature_importance.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------
# RESEARCH INTERPRETATION
# ---------------------------------------------------------------------
st.divider()
st.subheader("5 · What can we learn from this?")

top_names = importance.head(3)["Feature"].tolist()
top_name_text = ", ".join(pretty_feature(name) for name in top_names)

st.markdown(
    f"""
    **Main observations from this analysis**

    - The three highest-ranked features are **{top_name_text}**.
    - Current groundwater depth (`GWL_m_bgl`) has the largest mean absolute
      SHAP contribution in the reported analysis.
    - Historical mean groundwater depth and the month-related input also
      contribute substantially to the model output.
    - Previous groundwater observations contribute additional information
      about the station's recent history.
    - Geographic inputs have smaller global attribution values in this
      particular analysis than the leading temporal and groundwater-level
      features.

    **Important interpretation limits**

    - Mean absolute SHAP values measure attribution magnitude, not direction.
    - These values do not establish that a feature causes groundwater-level change.
    - Feature attribution can be affected by correlated inputs and the data
      used to fit and explain the model.
    - This ranking describes this fitted LightGBM model and the analysed
      validation sample; it is not a universal ranking of hydrogeological drivers.
    - SHAP explanations do not convert a predicted water level into an official
      groundwater-stress assessment.
    """
)

st.html(
    """
    <div class="shap-caveat">
        <strong>Local explanations are not currently displayed.</strong>
        A station-specific explanation requires observation-level SHAP values
        aligned with the corresponding input row. The currently referenced
        artefacts provide global feature importance and saved global plots only.
    </div>
    """
)