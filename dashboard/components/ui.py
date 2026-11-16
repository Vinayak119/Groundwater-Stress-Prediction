from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Optional

import streamlit as st

from dashboard.components.styles import inject_theme, status_pill


# ---------------------------------------------------------------------------
# Project paths and artifact availability
# ---------------------------------------------------------------------------

def project_root() -> Path:
    """Return the root directory of the groundwater research project."""
    return Path(__file__).resolve().parent.parent.parent


def _csv_count(path: Path) -> Optional[int]:
    """Count rows in a CSV without loading all columns into memory."""
    if not path.exists():
        return None

    try:
        import pandas as pd

        return len(pd.read_csv(path, usecols=[0]))
    except Exception:
        return None


def artifact_status() -> dict:
    """Return the availability and row counts of the research artifacts."""
    root = project_root()
    processed = root / "data" / "processed"
    metrics = root / "results" / "metrics"
    shap = root / "results" / "shap"
    models_dir = root / "models" / "trained"

    observations = processed / "observations_model_ready.csv"
    forecasting = processed / "forecasting_pairs_temporally_purged.csv"
    models = list(models_dir.glob("*.joblib")) if models_dir.exists() else []

    return {
        "observations_exists": observations.exists(),
        "observation_count": _csv_count(observations),
        "forecasting_exists": forecasting.exists(),
        "forecasting_count": _csv_count(forecasting),
        "model_count": len(models),
        "benchmark_exists": (metrics / "initial_model_benchmark.csv").exists(),
        "geographic_exists": (metrics / "geographic_holdout_results.csv").exists(),
        "shap_exists": (shap / "shap_feature_importance.csv").exists(),
        "intervals_exists": (metrics / "preliminary_prediction_intervals.csv").exists(),
    }


# ---------------------------------------------------------------------------
# Shared visual system
# ---------------------------------------------------------------------------

_SIDEBAR_CSS = """
<style>
/* Sidebar foundation */
[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 15% 0%, rgba(139,92,246,.13), transparent 34%),
        linear-gradient(180deg, #11111B 0%, #0D0D15 100%) !important;
    border-right: 1px solid rgba(167,139,250,.20) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: .65rem;
}

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding: .8rem .85rem 1.2rem;
}

/* Native multipage navigation */
[data-testid="stSidebarNav"] {
    padding-top: .35rem;
    margin-bottom: .8rem;
}

[data-testid="stSidebarNav"] ul {
    gap: .24rem;
}

[data-testid="stSidebarNav"] li {
    margin: 0 !important;
}

[data-testid="stSidebarNav"] a {
    border: 1px solid transparent !important;
    border-radius: 9px !important;
    padding: .58rem .72rem !important;
    margin: .08rem 0 !important;
    color: #C7C5D4 !important;
    font-family: 'DM Sans', 'Segoe UI', sans-serif !important;
    font-size: .86rem !important;
    font-weight: 550 !important;
    transition: background .18s ease, border-color .18s ease, color .18s ease;
}

[data-testid="stSidebarNav"] a:hover {
    background: rgba(167,139,250,.10) !important;
    border-color: rgba(167,139,250,.22) !important;
    color: #F5F3FF !important;
}

[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: linear-gradient(100deg, rgba(139,92,246,.24), rgba(96,165,250,.10)) !important;
    border: 1px solid rgba(167,139,250,.38) !important;
    color: #F5F3FF !important;
    box-shadow: inset 3px 0 0 #A78BFA, 0 4px 14px rgba(0,0,0,.12);
    font-weight: 700 !important;
}

/* Sidebar branding */
.gw-sidebar-brand {
    position: relative;
    overflow: hidden;
    padding: 1rem .95rem .95rem;
    margin: .45rem 0 1.05rem;
    border: 1px solid rgba(167,139,250,.25);
    border-radius: 15px;
    background:
        radial-gradient(circle at 90% 5%, rgba(52,211,153,.14), transparent 36%),
        linear-gradient(145deg, rgba(139,92,246,.16), rgba(25,25,40,.96) 62%);
    box-shadow: 0 9px 24px rgba(0,0,0,.18);
}

.gw-brand-mark {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 39px;
    height: 39px;
    margin-bottom: .72rem;
    border: 1px solid rgba(196,181,253,.42);
    border-radius: 12px;
    background: linear-gradient(145deg, rgba(167,139,250,.28), rgba(52,211,153,.12));
    font-size: 1.35rem;
    box-shadow: 0 4px 15px rgba(139,92,246,.13);
}

.gw-brand-name {
    color: #FAF9FF;
    font-family: 'Manrope', 'DM Sans', sans-serif;
    font-size: 1rem;
    line-height: 1.25;
    font-weight: 800;
    letter-spacing: -.025em;
}

.gw-brand-caption {
    margin-top: .35rem;
    color: #A9A6BB;
    font-family: 'DM Sans', sans-serif;
    font-size: .70rem;
    line-height: 1.5;
}

.gw-brand-tags {
    display: flex;
    flex-wrap: wrap;
    gap: .35rem;
    margin-top: .75rem;
}

.gw-brand-tag {
    padding: .20rem .48rem;
    border: 1px solid rgba(167,139,250,.24);
    border-radius: 20px;
    background: rgba(167,139,250,.09);
    color: #DCD3FF;
    font-family: 'DM Sans', sans-serif;
    font-size: .62rem;
    font-weight: 700;
    letter-spacing: .015em;
}

/* Sidebar section labels and status rows */
.gw-sidebar-section {
    margin: 1rem 0 .48rem;
    color: #AAA5C0;
    font-family: 'DM Sans', sans-serif;
    font-size: .66rem;
    font-weight: 800;
    letter-spacing: .13em;
    text-transform: uppercase;
}

.gw-pipeline-row {
    display: flex;
    align-items: flex-start;
    gap: .58rem;
    padding: .48rem .55rem;
    margin: .16rem 0;
    border: 1px solid transparent;
    border-radius: 9px;
    background: rgba(255,255,255,.018);
}

.gw-pipeline-dot {
    flex: 0 0 8px;
    width: 8px;
    height: 8px;
    margin-top: .32rem;
    border-radius: 50%;
    box-shadow: 0 0 10px currentColor;
}

.gw-pipeline-dot.ready {
    color: #34D399;
    background: #34D399;
}

.gw-pipeline-dot.pending {
    color: #F59E0B;
    background: #F59E0B;
}

.gw-pipeline-label {
    color: #D7D4E2;
    font-family: 'DM Sans', sans-serif;
    font-size: .77rem;
    line-height: 1.35;
    font-weight: 600;
}

.gw-pipeline-state {
    margin-top: .15rem;
    color: #858197;
    font-family: 'DM Sans', sans-serif;
    font-size: .66rem;
}

.gw-sidebar-data {
    padding: .72rem .75rem;
    border: 1px solid rgba(167,139,250,.18);
    border-radius: 11px;
    background: rgba(30,29,46,.72);
}

.gw-sidebar-data-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: .5rem;
    padding: .34rem 0;
    border-bottom: 1px solid rgba(255,255,255,.055);
}

.gw-sidebar-data-row:last-child {
    border-bottom: 0;
}

.gw-sidebar-data-label {
    color: #9F9BAF;
    font-family: 'DM Sans', sans-serif;
    font-size: .70rem;
}

.gw-sidebar-data-value {
    color: #F3F0FF;
    font-family: 'Manrope', sans-serif;
    font-size: .73rem;
    font-weight: 800;
    text-align: right;
}

.gw-sidebar-footer {
    margin-top: 1rem;
    padding: .7rem .2rem .1rem;
    border-top: 1px solid rgba(167,139,250,.16);
    color: #777387;
    font-family: 'DM Sans', sans-serif;
    font-size: .65rem;
    line-height: 1.55;
}

/* Consistent page typography and section spacing */
.gw-page-header {
    padding: .1rem 0 1.2rem;
    margin-bottom: 1.45rem;
    border-bottom: 1px solid rgba(167,139,250,.20);
}

.gw-eyebrow {
    color: #B9A5FF;
    font-size: .68rem;
    font-weight: 800;
    letter-spacing: .15em;
    text-transform: uppercase;
}

.gw-page-title {
    color: #FAF9FF;
    font-family: 'Manrope', sans-serif;
    font-size: 2rem;
    line-height: 1.2;
    font-weight: 800;
    letter-spacing: -.045em;
}

.gw-page-subtitle {
    color: #A9A6BB;
    font-size: .91rem;
    line-height: 1.6;
    margin-top: .45rem;
    max-width: 950px;
}

.section-header {
    color: #C4B5FD;
    font-size: .74rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .13em;
    border-bottom: 1px solid rgba(167,139,250,.18);
    padding-bottom: .68rem;
    margin: 1.55rem 0 1rem;
}

@media(max-width: 800px) {
    .gw-brand-name { font-size: .94rem; }
    .gw-page-title { font-size: 1.6rem; }
}
</style>
"""


def inject_sidebar_theme() -> None:
    """Apply the shared sidebar and navigation styling."""
    st.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page setup and reusable page components
# ---------------------------------------------------------------------------

def setup_page(
    title: str,
    icon: str = "💧",
    layout: str = "wide",
    subtitle: str = "",
) -> Path:
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        st.set_page_config(
            page_title=f"{title} · Groundwater Research",
            page_icon=icon,
            layout=layout,
            initial_sidebar_state="expanded",
        )
    except Exception:
        # Streamlit only allows page configuration once per page run.
        pass

    inject_theme()
    inject_sidebar_theme()
    sidebar_branding()
    sidebar_pipeline_status()
    data_status_sidebar()
    return root


def page_header(icon: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="gw-page-header">
          <div class="gw-eyebrow">Tamil Nadu · CGWB groundwater research</div>
          <div class="gw-page-title">{html.escape(str(icon))} &nbsp; {html.escape(str(title))}</div>
          <div class="gw-page-subtitle">{html.escape(str(subtitle))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar branding and live research pipeline
# ---------------------------------------------------------------------------

def sidebar_branding() -> None:
    st.sidebar.markdown(
        """
        <div class="gw-sidebar-brand">
          <div class="gw-brand-mark">💧</div>
          <div class="gw-brand-name">Groundwater Intelligence</div>
          <div class="gw-brand-caption">
            Tamil Nadu · CGWB observations<br>
            Research &amp; interpretable machine learning
          </div>
          <div class="gw-brand-tags">
            <span class="gw-brand-tag">RESEARCH</span>
            <span class="gw-brand-tag">TAMIL NADU</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_pipeline_status() -> None:
    s = artifact_status()
    phases = [
        ("Data audit & cleaning", s["observations_exists"]),
        ("Forecasting dataset", s["forecasting_exists"]),
        ("Trained model artifacts", s["model_count"] > 0),
        ("Temporal benchmark", s["benchmark_exists"]),
        ("Geographic evaluation", s["geographic_exists"]),
        ("SHAP analysis", s["shap_exists"]),
        ("Dashboard integration", True),
    ]

    st.sidebar.markdown(
        '<div class="gw-sidebar-section">Research pipeline</div>',
        unsafe_allow_html=True,
    )

    rows = []
    for label, ready in phases:
        state = "Available" if ready else "Not yet available"
        dot_class = "ready" if ready else "pending"
        rows.append(
            f"""
            <div class="gw-pipeline-row">
              <span class="gw-pipeline-dot {dot_class}"></span>
              <div>
                <div class="gw-pipeline-label">{html.escape(label)}</div>
                <div class="gw-pipeline-state">{state}</div>
              </div>
            </div>
            """
        )

    st.sidebar.markdown("".join(rows), unsafe_allow_html=True)


def data_status_sidebar() -> None:
    s = artifact_status()
    obs = f'{s["observation_count"]:,}' if s["observation_count"] is not None else "—"
    pairs = f'{s["forecasting_count"]:,}' if s["forecasting_count"] is not None else "—"
    models = f'{s["model_count"]:,}'

    st.sidebar.markdown(
        f"""
        <div class="gw-sidebar-section">Live project data</div>
        <div class="gw-sidebar-data">
          <div class="gw-sidebar-data-row">
            <span class="gw-sidebar-data-label">Clean observations</span>
            <span class="gw-sidebar-data-value">{obs}</span>
          </div>
          <div class="gw-sidebar-data-row">
            <span class="gw-sidebar-data-label">Forecasting pairs</span>
            <span class="gw-sidebar-data-value">{pairs}</span>
          </div>
          <div class="gw-sidebar-data-row">
            <span class="gw-sidebar-data-label">Saved models</span>
            <span class="gw-sidebar-data-value">{models}</span>
          </div>
          <div class="gw-sidebar-data-row">
            <span class="gw-sidebar-data-label">Study region</span>
            <span class="gw-sidebar-data-value">Tamil Nadu</span>
          </div>
          <div class="gw-sidebar-data-row">
            <span class="gw-sidebar-data-label">Forecast horizon</span>
            <span class="gw-sidebar-data-value">≤ 150 days</span>
          </div>
        </div>
        <div class="gw-sidebar-footer">
          CGWB groundwater research dashboard<br>
          Local research environment · Artifact-backed results
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Shared metrics, cards, messages and data display
# ---------------------------------------------------------------------------

def metric_row(metrics: list[dict]) -> None:
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.metric(
                label=m["label"],
                value=m["value"],
                delta=m.get("delta"),
                delta_color=m.get("delta_color", "normal"),
                help=m.get("help"),
            )


def kpi_card(
    icon: str,
    title: str,
    value: str,
    subtitle: str = "",
    status_tag: str = "",
    border_color: str = "#39304F",
) -> None:
    # border_color is retained for compatibility with existing page calls.
    safe_border = html.escape(str(border_color), quote=True)
    st.markdown(
        f"""
        <div class="gw-card" style="border-top:2px solid {safe_border}">
          <div style="font-size:1.35rem">{html.escape(str(icon))}</div>
          <div class="gw-card-title">{html.escape(str(title))}</div>
          <div class="gw-card-value">{html.escape(str(value))}</div>
          <div class="gw-card-sub">{html.escape(str(subtitle))}</div>
          <div style="margin-top:.45rem">{status_tag}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def no_data_state(
    icon: str = "🗂️",
    title: str = "No data available",
    body: str = "The required research artifact has not been generated.",
    action: str = "",
) -> None:
    st.info(f"{icon} **{title}**\n\n{body}\n\n{action}")


def model_unavailable_state() -> None:
    model_dir = project_root() / "models" / "trained"
    no_data_state(
        icon="🤖",
        title="No trained models found",
        body=f"Expected saved model files in {model_dir}",
        action="Train and save a model before using inference.",
    )


def pending_badge(label: str = "Pending real data") -> str:
    return status_pill(label, "pending")


def success_badge(label: str = "Ready") -> str:
    return status_pill(label, "ready")


def demo_badge(label: str = "Preview") -> str:
    return status_pill(label, "demo")


def styled_dataframe(
    df,
    height: int = 400,
    use_container_width: bool = True,
) -> None:
    st.dataframe(
        df,
        height=height,
        use_container_width=use_container_width,
        hide_index=True,
    )
