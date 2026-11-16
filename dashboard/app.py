
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.components.ui import setup_page
from dashboard.components.styles import PLOTLY_TEMPLATE, section_header

root = setup_page(title="Groundwater Intelligence", icon="💧")


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def read_csv(relative_path):
    path = root / relative_path

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def read_json(relative_path):
    path = root / relative_path

    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


observations = read_csv(
    "data/processed/observations_model_ready.csv"
)

pairs = read_csv(
    "data/processed/forecasting_pairs_temporally_purged.csv"
)

benchmark = read_csv(
    "results/metrics/initial_model_benchmark.csv"
)

geo = read_csv(
    "results/metrics/geographic_holdout_results.csv"
)

districts = read_csv(
    "results/metrics/geographic_district_performance.csv"
)

shap_data = read_csv(
    "results/shap/shap_feature_importance.csv"
)

intervals = read_csv(
    "results/metrics/preliminary_prediction_intervals.csv"
)

model_dir = root / "models" / "trained"

model_count = (
    len(list(model_dir.glob("*.joblib")))
    if model_dir.exists()
    else 0
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_number(value, digits=2):
    try:
        number = float(value)

        if not np.isfinite(number):
            return "Unavailable"

        return f"{number:,.{digits}f}"

    except (TypeError, ValueError):
        return "Unavailable"


def html_card(label, value, description="", accent="cyan"):

    colors = {
        "cyan": "#A78BFA",
        "blue": "#60A5FA",
        "green": "#34D399",
        "amber": "#F59E0B",
    }

    color = colors.get(accent, colors["cyan"])

    st.html(
        f"""
        <div class="home-kpi">
            <div class="home-kpi-indicator"
                 style="background:{color}"></div>

            <div class="home-kpi-label">
                {html.escape(str(label))}
            </div>

            <div class="home-kpi-value">
                {html.escape(str(value))}
            </div>

            <div class="home-kpi-description">
                {html.escape(str(description))}
            </div>
        </div>
        """
    )


def page_link_card(page, icon, title, description):

    st.html(
        f"""
        <div class="module-card">
            <div class="module-icon">{icon}</div>

            <div class="module-title">
                {html.escape(title)}
            </div>

            <div class="module-description">
                {html.escape(description)}
            </div>
        </div>
        """
    )

    st.page_link(
        page,
        label=f"Open {title}",
        icon="↗",
    )


# ============================================================
# CUSTOM HOME PAGE STYLING
# ============================================================

st.html(
    """
    <style>

    .main .block-container {
        max-width: 1500px;
        padding-top: 1.8rem;
    }

    .home-hero {
        position: relative;
        overflow: hidden;

        background:
            radial-gradient(
                circle at 88% 15%,
                rgba(167,139,250,.20),
                transparent 34%
            ),
            radial-gradient(circle at 12% 100%, rgba(52,211,153,.09), transparent 35%),
            linear-gradient(120deg, #242039, #151522 72%);

        border: 1px solid #39304F;
        border-radius: 19px;

        padding: 2.4rem;
        margin: .3rem 0 1.8rem;
    }

    .home-eyebrow {
        color: #A78BFA;
        font-size: .73rem;
        font-weight: 750;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: .85rem;
    }

    .home-title {
        color: #F8FAFC;
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -.05em;
        max-width: 850px;
    }

    .home-description {
        color: #D4D4DC;
        font-size: .98rem;
        line-height: 1.75;
        max-width: 780px;
        margin-top: .9rem;
    }

    .home-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 9px;
        margin-top: 1.3rem;
    }

    .home-meta span {
        background: rgba(167,139,250,.12);
        border: 1px solid rgba(167,139,250,.35);
        color: #DDD6FE;
        padding: 6px 11px;
        border-radius: 20px;
        font-size: .72rem;
        font-weight: 600;
    }

    .home-kpi {
        position: relative;
        background: linear-gradient(145deg,#242039,#181824);
        border: 1px solid #39304F;
        border-radius: 14px;
        padding: 1.2rem 1.25rem;
        min-height: 137px;
        margin-bottom: .8rem;
        overflow: hidden;
        transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
    }

    .home-kpi:hover {
        transform: translateY(-3px);
        border-color: #A78BFA;
        box-shadow: 0 12px 28px rgba(0,0,0,.24);
    }

    .home-kpi-indicator {
        width: 28px;
        height: 3px;
        border-radius: 5px;
        margin-bottom: 13px;
    }

    .home-kpi-label {
        color: #B4B4C2;
        font-size: .71rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .075em;
    }

    .home-kpi-value {
        color: #F8FAFC;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -.045em;
        margin-top: 5px;
    }

    .home-kpi-description {
        color: #A1A1AA;
        font-size: .75rem;
        margin-top: 3px;
        line-height: 1.45;
    }

    .home-panel {
        background: #181824;
        border: 1px solid #39304F;
        border-radius: 14px;
        padding: 1.2rem 1.3rem;
        margin-bottom: .9rem;
    }

    .home-panel-title {
        color: #F8FAFC;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: .3rem;
    }

    .home-panel-subtitle {
        color: #A1A1AA;
        font-size: .79rem;
        line-height: 1.55;
    }

    .home-notice {
        background: #242039;
        border: 1px solid #51416E;
        border-left: 4px solid #A78BFA;
        border-radius: 10px;
        padding: 1rem 1.15rem;
        color: #DDD6FE;
        font-size: .84rem;
        line-height: 1.65;
        margin: .7rem 0 1rem;
    }

    .home-warning {
        background: #332719;
        border: 1px solid #80602D;
        border-left: 4px solid #F4BE65;
        border-radius: 10px;
        padding: 1rem 1.15rem;
        color: #FDE68A;
        font-size: .84rem;
        line-height: 1.65;
        margin: .8rem 0;
    }

    .module-card {
        background: linear-gradient(145deg,#242039,#181824);
        border: 1px solid #39304F;
        border-radius: 13px;
        padding: 1.15rem;
        min-height: 148px;
        margin-bottom: .2rem;
    }

    .module-icon {
        font-size: 1.45rem;
        margin-bottom: .7rem;
    }

    .module-title {
        color: #F8FAFC;
        font-size: .96rem;
        font-weight: 700;
        margin-bottom: .4rem;
    }

    .module-description {
        color: #B4B4C2;
        font-size: .79rem;
        line-height: 1.55;
    }

    .home-footer {
        color: #85859A;
        border-top: 1px solid #39304F;
        padding-top: 1.1rem;
        margin-top: 2rem;
        font-size: .76rem;
        line-height: 1.6;
    }

    @media(max-width:800px) {

        .home-hero {
            padding: 1.5rem;
        }

        .home-kpi-value {
            font-size: 1.6rem;
        }

    }

    </style>
    """
)


# ============================================================
# HERO SECTION
# ============================================================

st.html(
    """
    <div class="home-hero">

        <div class="home-eyebrow">
            TAMIL NADU · GROUNDWATER RESEARCH PLATFORM
        </div>

        <div class="home-title">
            Understanding groundwater.<br>
            Predicting what comes next.
        </div>

        <div class="home-description">
            An analytical research platform built around CGWB
            groundwater observations, temporal forecasting,
            geographic evaluation and interpretable machine learning.
        </div>

        <div class="home-meta">
            <span>CGWB observations</span>
            <span>Temporal modelling</span>
            <span>Geographic evaluation</span>
            <span>Explainable AI</span>
        </div>

    </div>
    """
)


st.html(
    """
    <div class="home-notice">
        <strong>Research mode active.</strong>
        This dashboard displays actual project artifacts.
        Missing files are reported as unavailable.
        Synthetic demonstration results are not substituted.
    </div>
    """
)


# ============================================================
# RESEARCH AT A GLANCE
# ============================================================

section_header("Research at a glance")

observation_count = (
    f"{len(observations):,}"
    if observations is not None
    else "Unavailable"
)

pair_count = (
    f"{len(pairs):,}"
    if pairs is not None
    else "Unavailable"
)

shap_count = (
    f"{len(shap_data):,}"
    if shap_data is not None
    else "Unavailable"
)

district_count = (
    f"{observations['District'].nunique():,}"
    if observations is not None and "District" in observations.columns
    else "Unavailable"
)

k1, k2, k3, k4 = st.columns(4)

with k1:
    html_card(
        "Clean observations",
        observation_count,
        "Retained groundwater measurements",
        "cyan",
    )

with k2:
    html_card(
        "Forecasting pairs",
        pair_count,
        "Temporally purged next-observation pairs",
        "blue",
    )

with k3:
    html_card(
        "Trained model files",
        f"{model_count:,}",
        "Saved model artifacts",
        "green",
    )

with k4:
    html_card(
        "Districts represented",
        district_count,
        "Districts in cleaned observations",
        "amber",
    )

st.caption(
    f"SHAP importance records available: {shap_count}. "
    "Forecasting pairs represent the next available observation "
    "within the defined 150-day horizon, not fixed 90-day forecasts."
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

section_header("Dataset overview")

if observations is not None:

    date_col = "Parsed_Date"
    value_col = "GWL_m_bgl"
    station_col = "Provisional_Station_ID"

    dates = (
        pd.to_datetime(
            observations[date_col],
            errors="coerce"
        ).dropna()
        if date_col in observations.columns
        else pd.Series(dtype="datetime64[ns]")
    )

    values = (
        pd.to_numeric(
            observations[value_col],
            errors="coerce"
        )
        if value_col in observations.columns
        else pd.Series(dtype=float)
    )

    stations = (
        observations[station_col].nunique()
        if station_col in observations.columns
        else None
    )

    missing_values = (
        int(values.isna().sum())
        if value_col in observations.columns
        else None
    )

    left, right = st.columns([1.2, 1])

    with left:

        st.html(
            """
            <div class="home-panel">
                <div class="home-panel-title">
                    Observation coverage
                </div>
                <div class="home-panel-subtitle">
                    Historical extent of the cleaned observation dataset
                </div>
            </div>
            """
        )

        x1, x2 = st.columns(2)

        with x1:
            html_card(
                "First observation",
                f"{dates.min():%Y-%m-%d}" if not dates.empty else "Unavailable",
                "Earliest retained date",
                "blue",
            )

        with x2:
            html_card(
                "Latest observation",
                f"{dates.max():%Y-%m-%d}" if not dates.empty else "Unavailable",
                "Latest retained date",
                "cyan",
            )

        x3, x4 = st.columns(2)

        with x3:
            html_card(
                "Provisional station IDs",
                f"{stations:,}" if stations is not None else "Unavailable",
                "Not verified official well identities",
                "green",
            )

        with x4:
            html_card(
                "Missing GWL values",
                f"{missing_values:,}" if missing_values is not None else "Unavailable",
                "In cleaned observation data",
                "amber",
            )

    with right:

        st.html(
            """
            <div class="home-panel">
                <div class="home-panel-title">
                    Observation activity
                </div>
                <div class="home-panel-subtitle">
                    Monthly count of retained measurements
                </div>
            </div>
            """
        )

        if not dates.empty:

            # Count observations by month and explicitly retain months
            # with no records so gaps are visible rather than skipped.
            monthly_counts = dates.dt.to_period("M").value_counts().sort_index()

            month_index = pd.period_range(
                start=dates.min().to_period("M"),
                end=dates.max().to_period("M"),
                freq="M",
            )

            monthly_counts = monthly_counts.reindex(
                month_index,
                fill_value=0,
            )

            monthly_df = pd.DataFrame({
                "Month": month_index.to_timestamp(),
                "Observations": monthly_counts.to_numpy(),
            })

            # Descriptive smoothing only; this is not a forecast.
            monthly_df["12-month average"] = (
                monthly_df["Observations"]
                .rolling(window=12, min_periods=1)
                .mean()
            )

            fig = px.line(
                monthly_df,
                x="Month",
                y="Observations",
                template=PLOTLY_TEMPLATE,
            )

            fig.update_traces(
                line=dict(color="#A78BFA", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(167,139,250,0.10)",
                name="Monthly observations",
                hovertemplate=(
                    "%{x|%b %Y}<br>"
                    "Records: %{y:,}<extra></extra>"
                ),
            )

            fig.add_scatter(
                x=monthly_df["Month"],
                y=monthly_df["12-month average"],
                mode="lines",
                name="12-month average",
                line=dict(color="#34D399", width=2.5),
                hovertemplate=(
                    "12-month average: %{y:.1f}<extra></extra>"
                ),
            )

            fig.update_layout(
                height=340,
                showlegend=True,
                legend=dict(orientation="h", y=1.12, x=0),
                xaxis_title="Year",
                yaxis_title="Monthly observations",
                hovermode="x unified",
                margin=dict(t=35, b=45, l=35, r=15),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:
            st.warning("Observation dates are unavailable.")

    st.html(
        """
        <div class="home-warning">
            <strong>Data interpretation:</strong>
            Station identifiers are provisional dataset identifiers,
            not verified official well identities. Unresolved identity
            conflicts and flagged anomalies require careful interpretation.
            Groundwater depth in metres below ground level is not, by itself,
            an official groundwater-stress classification.
        </div>
        """
    )

else:
    st.warning(
        "The cleaned observation dataset is unavailable. "
        "Dataset overview cannot be displayed."
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

section_header("Research model evaluation")

if benchmark is not None and not benchmark.empty:
    required_columns = {"Model", "Split", "MAE", "RMSE", "R2"}
    if required_columns.issubset(benchmark.columns):
        benchmark_view = benchmark.copy()
        benchmark_view["Split"] = (
            benchmark_view["Split"].astype(str).str.strip().str.lower()
        )
        for metric in ["MAE", "RMSE", "R2"]:
            benchmark_view[metric] = pd.to_numeric(
                benchmark_view[metric], errors="coerce"
            )

        validation = benchmark_view[
            benchmark_view["Split"] == "validation"
        ].copy()
        test = benchmark_view[
            benchmark_view["Split"] == "test"
        ].copy()

        best_validation = (
            validation.loc[validation["MAE"].idxmin()]
            if not validation.empty and validation["MAE"].notna().any()
            else None
        )
        best_test = (
            test.loc[test["MAE"].idxmin()]
            if not test.empty and test["MAE"].notna().any()
            else None
        )
        best_test_rmse = (
            test.loc[test["RMSE"].idxmin()]
            if not test.empty and test["RMSE"].notna().any()
            else None
        )

        st.caption(
            "Temporal evaluation uses a time-ordered validation and test split. "
            "MAE and RMSE are in metres of groundwater depth (m bgl); "
            "lower is better. R² is unitless; higher is better."
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            html_card(
                "Best validation MAE",
                (
                    f'{safe_number(best_validation["MAE"])} m'
                    if best_validation is not None else "Unavailable"
                ),
                (
                    str(best_validation["Model"])
                    if best_validation is not None else "No validation result"
                ),
                "blue",
            )
        with m2:
            html_card(
                "Best test MAE",
                (
                    f'{safe_number(best_test["MAE"])} m'
                    if best_test is not None else "Unavailable"
                ),
                (
                    str(best_test["Model"])
                    if best_test is not None else "No test result"
                ),
                "green",
            )
        with m3:
            html_card(
                "Lowest test RMSE",
                (
                    f'{safe_number(best_test_rmse["RMSE"])} m'
                    if best_test_rmse is not None else "Unavailable"
                ),
                (
                    str(best_test_rmse["Model"])
                    if best_test_rmse is not None else "No test result"
                ),
                "amber",
            )

        chart_left, chart_right = st.columns(2)

        with chart_left:
            st.html(
                """
                <div class="home-panel">
                    <div class="home-panel-title">MAE comparison</div>
                    <div class="home-panel-subtitle">
                        Validation versus held-out temporal test set
                    </div>
                </div>
                """
            )
            mae_long = benchmark_view.dropna(
                subset=["Model", "Split", "MAE"]
            )
            if not mae_long.empty:
                mae_fig = px.bar(
                    mae_long,
                    x="Model",
                    y="MAE",
                    color="Split",
                    barmode="group",
                    color_discrete_map={
                        "validation": "#A78BFA",
                        "test": "#34D399",
                    },
                    template=PLOTLY_TEMPLATE,
                    text_auto=".2f",
                )
                mae_fig.update_layout(
                    height=370,
                    yaxis_title="MAE (m)",
                    xaxis_title="",
                    legend_title="Split",
                    margin=dict(t=20, b=90, l=35, r=15),
                )
                mae_fig.update_xaxes(tickangle=-20)
                st.plotly_chart(mae_fig, use_container_width=True)
            else:
                st.info("MAE values are unavailable.")

        with chart_right:
            st.html(
                """
                <div class="home-panel">
                    <div class="home-panel-title">RMSE comparison</div>
                    <div class="home-panel-subtitle">
                        Penalises larger forecast errors more strongly
                    </div>
                </div>
                """
            )
            rmse_long = benchmark_view.dropna(
                subset=["Model", "Split", "RMSE"]
            )
            if not rmse_long.empty:
                rmse_fig = px.bar(
                    rmse_long,
                    x="Model",
                    y="RMSE",
                    color="Split",
                    barmode="group",
                    color_discrete_map={
                        "validation": "#A78BFA",
                        "test": "#34D399",
                    },
                    template=PLOTLY_TEMPLATE,
                    text_auto=".2f",
                )
                rmse_fig.update_layout(
                    height=370,
                    yaxis_title="RMSE (m)",
                    xaxis_title="",
                    legend_title="Split",
                    margin=dict(t=20, b=90, l=35, r=15),
                )
                rmse_fig.update_xaxes(tickangle=-20)
                st.plotly_chart(rmse_fig, use_container_width=True)
            else:
                st.info("RMSE values are unavailable.")

        with st.expander("View complete temporal benchmark table"):
            display_benchmark = benchmark_view.copy()
            display_benchmark["MAE"] = display_benchmark["MAE"].round(3)
            display_benchmark["RMSE"] = display_benchmark["RMSE"].round(3)
            display_benchmark["R2"] = display_benchmark["R2"].round(3)
            st.dataframe(
                display_benchmark,
                use_container_width=True,
                hide_index=True,
            )

        st.html(
            """
            <div class="home-warning">
                <strong>Interpretation:</strong>
                The persistence baseline has the lowest test MAE and RMSE
                in this initial temporal benchmark. LightGBM has the lowest
                validation MAE among the listed models, but it does not
                outperform persistence on the temporal test set. Model
                selection should therefore not be based on validation scores
                alone.
            </div>
            """
        )
    else:
        st.warning(
            "The temporal benchmark file is missing required columns: "
            "Model, Split, MAE, RMSE and R2."
        )
else:
    st.warning("Initial model benchmark is unavailable.")

left, right = st.columns(2)

with left:
    st.html(
        """
        <div class="home-panel">
            <div class="home-panel-title">R² comparison</div>
            <div class="home-panel-subtitle">
                Proportion of target variance explained on each temporal split
            </div>
        </div>
        """
    )
    if benchmark is not None and {"Model", "Split", "R2"}.issubset(benchmark.columns):
        r2_data = benchmark.copy()
        r2_data["R2"] = pd.to_numeric(r2_data["R2"], errors="coerce")
        r2_data = r2_data.dropna(subset=["R2"])
        if not r2_data.empty:
            r2_fig = px.bar(
                r2_data,
                x="Model",
                y="R2",
                color="Split",
                barmode="group",
                color_discrete_map={
                    "validation": "#A78BFA",
                    "test": "#34D399",
                },
                template=PLOTLY_TEMPLATE,
                text_auto=".3f",
            )
            r2_fig.update_layout(
                height=350,
                yaxis_title="R²",
                xaxis_title="",
                legend_title="Split",
                margin=dict(t=15, b=90, l=35, r=15),
            )
            r2_fig.update_xaxes(tickangle=-20)
            st.plotly_chart(r2_fig, use_container_width=True)
        else:
            st.info("R² values are unavailable.")
    else:
        st.info("R² benchmark data is unavailable.")

with right:
    st.html(
        """
        <div class="home-panel">
            <div class="home-panel-title">Geographic holdout</div>
            <div class="home-panel-subtitle">
                Aggregate performance in districts withheld from model training
            </div>
        </div>
        """
    )
    if geo is not None and not geo.empty:
        geo_view = geo.copy()
        for metric in ["MAE", "RMSE", "R2"]:
            if metric in geo_view.columns:
                geo_view[metric] = pd.to_numeric(
                    geo_view[metric], errors="coerce"
                )

        if {"Model", "MAE", "RMSE", "R2"}.issubset(geo_view.columns):
            best_geo_mae = geo_view.loc[geo_view["MAE"].idxmin()]
            best_geo_rmse = geo_view.loc[geo_view["RMSE"].idxmin()]

            g1, g2 = st.columns(2)
            with g1:
                html_card(
                    "Lowest geographic MAE",
                    f'{safe_number(best_geo_mae["MAE"])} m',
                    str(best_geo_mae["Model"]),
                    "green",
                )
            with g2:
                html_card(
                    "Lowest geographic RMSE",
                    f'{safe_number(best_geo_rmse["RMSE"])} m',
                    str(best_geo_rmse["Model"]),
                    "blue",
                )

            geo_display = geo_view.copy()
            for metric in ["MAE", "RMSE", "R2"]:
                geo_display[metric] = geo_display[metric].round(3)

            st.dataframe(
                geo_display,
                use_container_width=True,
                hide_index=True,
            )

            if (
                "LightGBM" in geo_view["Model"].astype(str).values
                and "Persistence" in geo_view["Model"].astype(str).values
            ):
                lgbm_row = geo_view[
                    geo_view["Model"].astype(str).str.contains(
                        "LightGBM", case=False, na=False
                    )
                ].iloc[0]
                persistence_row = geo_view[
                    geo_view["Model"].astype(str).str.contains(
                        "Persistence", case=False, na=False
                    )
                ].iloc[0]

                mae_gain = (
                    (persistence_row["MAE"] - lgbm_row["MAE"])
                    / persistence_row["MAE"] * 100
                    if pd.notna(persistence_row["MAE"])
                    and persistence_row["MAE"] != 0
                    else np.nan
                )
                rmse_gain = (
                    (persistence_row["RMSE"] - lgbm_row["RMSE"])
                    / persistence_row["RMSE"] * 100
                    if pd.notna(persistence_row["RMSE"])
                    and persistence_row["RMSE"] != 0
                    else np.nan
                )

                st.caption(
                    "Compared with persistence, LightGBM reduces geographic "
                    f"holdout MAE by {safe_number(mae_gain, 1)}% and RMSE by "
                    f"{safe_number(rmse_gain, 1)}% on this evaluation."
                )

            st.caption(
                "This is a district holdout within Tamil Nadu, "
                "not independent cross-state validation."
            )
        else:
            st.warning(
                "Geographic holdout data must contain Model, MAE, RMSE and R2."
            )
    else:
        st.warning("Geographic holdout results are unavailable.")


# ============================================================
# DISTRICT-LEVEL GEOGRAPHIC GENERALISATION
# ============================================================

section_header("District-level geographic performance")

if districts is not None and not districts.empty:
    district_view = districts.copy()
    required = {"District", "Stations", "Predictions", "MAE", "RMSE"}

    if required.issubset(district_view.columns):
        for metric in ["Stations", "Predictions", "MAE", "RMSE"]:
            district_view[metric] = pd.to_numeric(
                district_view[metric], errors="coerce"
            )

        district_view = district_view.dropna(
            subset=["District", "MAE", "RMSE"]
        )

        if not district_view.empty:
            best_mae_district = district_view.loc[district_view["MAE"].idxmin()]
            worst_mae_district = district_view.loc[district_view["MAE"].idxmax()]
            worst_rmse_district = district_view.loc[district_view["RMSE"].idxmax()]

            d1, d2, d3 = st.columns(3)
            with d1:
                html_card(
                    "Lowest district MAE",
                    f'{safe_number(best_mae_district["MAE"])} m',
                    str(best_mae_district["District"]),
                    "green",
                )
            with d2:
                html_card(
                    "Highest district MAE",
                    f'{safe_number(worst_mae_district["MAE"])} m',
                    str(worst_mae_district["District"]),
                    "amber",
                )
            with d3:
                html_card(
                    "Highest district RMSE",
                    f'{safe_number(worst_rmse_district["RMSE"])} m',
                    str(worst_rmse_district["District"]),
                    "amber",
                )

            district_left, district_right = st.columns(2)

            with district_left:
                st.html(
                    """
                    <div class="home-panel">
                        <div class="home-panel-title">District-wise MAE</div>
                        <div class="home-panel-subtitle">
                            Mean absolute error; lower values indicate better
                            average accuracy
                        </div>
                    </div>
                    """
                )
                mae_district = district_view.sort_values("MAE", ascending=True)
                mae_fig = px.bar(
                    mae_district,
                    x="MAE",
                    y="District",
                    orientation="h",
                    template=PLOTLY_TEMPLATE,
                    text="MAE",
                )
                mae_fig.update_traces(
                    marker_color="#A78BFA",
                    texttemplate="%{x:.2f}",
                    textposition="outside",
                    hovertemplate=(
                        "%{y}<br>MAE: %{x:.3f} m<extra></extra>"
                    ),
                )
                mae_fig.update_layout(
                    height=390,
                    xaxis_title="MAE (m)",
                    yaxis_title="",
                    showlegend=False,
                    margin=dict(t=15, b=35, l=15, r=45),
                )
                st.plotly_chart(mae_fig, use_container_width=True)

            with district_right:
                st.html(
                    """
                    <div class="home-panel">
                        <div class="home-panel-title">District-wise RMSE</div>
                        <div class="home-panel-subtitle">
                            Root mean squared error; more sensitive to large
                            individual errors
                        </div>
                    </div>
                    """
                )
                rmse_district = district_view.sort_values(
                    "RMSE", ascending=True
                )
                rmse_fig = px.bar(
                    rmse_district,
                    x="RMSE",
                    y="District",
                    orientation="h",
                    template=PLOTLY_TEMPLATE,
                    text="RMSE",
                )
                rmse_fig.update_traces(
                    marker_color="#34D399",
                    texttemplate="%{x:.2f}",
                    textposition="outside",
                    hovertemplate=(
                        "%{y}<br>RMSE: %{x:.3f} m<extra></extra>"
                    ),
                )
                rmse_fig.update_layout(
                    height=390,
                    xaxis_title="RMSE (m)",
                    yaxis_title="",
                    showlegend=False,
                    margin=dict(t=15, b=35, l=15, r=45),
                )
                st.plotly_chart(rmse_fig, use_container_width=True)

            with st.expander("Inspect district sample sizes and metrics"):
                district_display = district_view.sort_values(
                    "MAE", ascending=True
                ).copy()
                for metric in ["MAE", "RMSE"]:
                    district_display[metric] = district_display[metric].round(3)
                st.dataframe(
                    district_display,
                    use_container_width=True,
                    hide_index=True,
                )

            st.html(
                """
                <div class="home-warning">
                    <strong>Research interpretation:</strong>
                    District-level metrics are based on different numbers of
                    stations and predictions. RMSE can rise sharply when a
                    district contains a small number of large errors. These
                    results describe this held-out Tamil Nadu evaluation and
                    do not establish performance in other states or aquifers.
                </div>
                """
            )
        else:
            st.warning("District-level metrics contain no usable records.")
    else:
        st.warning(
            "District-level results must contain District, Stations, "
            "Predictions, MAE and RMSE columns."
        )
else:
    st.warning("District-level geographic performance is unavailable.")



# ============================================================
# RESEARCH MODULES
# ============================================================

section_header("Explore research modules")

modules = [
    (
        "pages/1_Geographic_Analysis.py",
        "🗺️",
        "Geographic Analysis",
        "Explore station locations, districts and spatial patterns.",
    ),
    (
        "pages/2_Data_Explorer.py",
        "🔎",
        "Data Explorer",
        "Inspect groundwater measurements and data quality.",
    ),
    (
        "pages/3_Groundwater_Prediction.py",
        "💧",
        "Groundwater Prediction",
        "Generate next-observation forecasts using saved models.",
    ),
    (
        "pages/4_Model_Performance.py",
        "📊",
        "Model Performance",
        "Compare temporal and geographic model performance.",
    ),
    (
        "pages/5_Explainable_AI.py",
        "🧠",
        "Explainable AI",
        "Explore SHAP feature attribution and model behaviour.",
    ),
    (
        "pages/6_Research_Insights.py",
        "📚",
        "Research Insights",
        "Review research methodology, evidence and limitations.",
    ),
]

for start in range(0, len(modules), 3):

    cols = st.columns(3)

    for col, module in zip(
        cols,
        modules[start:start + 3]
    ):

        with col:
            page_link_card(*module)


# ============================================================
# ADDITIONAL RESEARCH ARTIFACTS
# ============================================================

with st.expander("View district-level evaluation"):

    if districts is not None:

        st.dataframe(
            districts,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.warning("District-level evaluation is unavailable.")


with st.expander("Explore SHAP feature importance", expanded=False):

    if shap_data is not None and not shap_data.empty:
        shap_view = shap_data.copy()

        if {"Feature", "Importance_Percent"}.issubset(shap_view.columns):
            shap_view["Importance_Percent"] = pd.to_numeric(
                shap_view["Importance_Percent"], errors="coerce"
            )
            shap_view = shap_view.dropna(
                subset=["Feature", "Importance_Percent"]
            ).sort_values("Importance_Percent", ascending=False)

            if not shap_view.empty:
                top_feature = shap_view.iloc[0]
                top_three_share = shap_view.head(3)["Importance_Percent"].sum()

                s1, s2 = st.columns(2)
                with s1:
                    html_card(
                        "Most influential feature",
                        f'{safe_number(top_feature["Importance_Percent"], 1)}%',
                        str(top_feature["Feature"]).replace("_", " "),
                        "blue",
                    )
                with s2:
                    html_card(
                        "Top three feature share",
                        f"{safe_number(top_three_share, 1)}%",
                        "Combined mean absolute SHAP importance",
                        "green",
                    )

                shap_plot_data = shap_view.head(12).sort_values(
                    "Importance_Percent", ascending=True
                ).copy()
                shap_plot_data["Feature label"] = (
                    shap_plot_data["Feature"]
                    .astype(str)
                    .str.replace("_", " ", regex=False)
                )

                shap_fig = px.bar(
                    shap_plot_data,
                    x="Importance_Percent",
                    y="Feature label",
                    orientation="h",
                    template=PLOTLY_TEMPLATE,
                    text="Importance_Percent",
                )
                shap_fig.update_traces(
                    marker_color="#A78BFA",
                    texttemplate="%{text:.1f}%",
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="%{y}<br>Importance: %{x:.2f}%<extra></extra>",
                )
                shap_fig.update_layout(
                    height=470,
                    showlegend=False,
                    xaxis_title="Mean absolute SHAP importance (%)",
                    yaxis_title="",
                    margin=dict(t=15, b=45, l=15, r=55),
                )
                st.plotly_chart(shap_fig, use_container_width=True)

                shap_display = shap_view.copy()
                shap_display["Importance_Percent"] = (
                    shap_display["Importance_Percent"].round(3)
                )
                if "Mean_Absolute_SHAP" in shap_display.columns:
                    shap_display["Mean_Absolute_SHAP"] = pd.to_numeric(
                        shap_display["Mean_Absolute_SHAP"], errors="coerce"
                    ).round(5)

                with st.expander("View full SHAP importance table"):
                    st.dataframe(
                        shap_display,
                        use_container_width=True,
                        hide_index=True,
                    )

                st.caption(
                    "The leading features are current groundwater depth, "
                    "historical mean groundwater depth and current month. "
                    "Together, the top three account for about 75.2% of the "
                    "reported mean absolute SHAP importance."
                )
                st.warning(
                    "SHAP importance describes how features contribute to "
                    "the fitted model's predictions. It does not establish "
                    "causality, physical groundwater drivers or feature "
                    "effects in every individual prediction."
                )
            else:
                st.warning("SHAP importance values are unavailable.")
        else:
            st.warning(
                "SHAP data must contain Feature and Importance_Percent columns."
            )
    else:
        st.warning("SHAP feature importance is unavailable.")


with st.expander("View preliminary prediction intervals"):

    if intervals is not None:

        st.dataframe(
            intervals,
            use_container_width=True,
            hide_index=True,
        )

        st.warning(
            "These intervals are preliminary. Calibration uses validation "
            "data involved in model selection and should not be described "
            "as final independently calibrated uncertainty."
        )

    else:
        st.warning("Prediction interval results are unavailable.")


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="home-footer">
        <strong>Groundwater Research Intelligence</strong>
        <br>
        Academic research dashboard · CGWB Tamil Nadu groundwater observations

        <br><br>

        Research interpretation should account for provisional station
        identities, unresolved data anomalies, temporal validation limits,
        geographic generalisation and model uncertainty.
    </div>
    """
)