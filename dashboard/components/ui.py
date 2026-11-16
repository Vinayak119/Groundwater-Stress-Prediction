from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
import html
import streamlit as st

from dashboard.components.styles import inject_theme, status_pill


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def artifact_status() -> dict:
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


def _csv_count(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        import pandas as pd
        return len(pd.read_csv(path, usecols=[0]))
    except Exception:
        return None


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
        pass

    inject_theme()
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


def sidebar_branding() -> None:
    st.sidebar.markdown(
        """
        <div style="padding:.45rem .1rem 1rem;border-bottom:1px solid #26364D;margin-bottom:1rem">
          <div style="font-size:1.7rem">💧</div>
          <div style="font-size:1rem;font-weight:800;color:#EAF1FA">Groundwater Research</div>
          <div style="font-size:.72rem;color:#91A5BF;margin-top:.2rem">Tamil Nadu · CGWB · Interpretable ML</div>
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
    st.sidebar.markdown("**Research pipeline**")
    for label, ready in phases:
        dot = "🟢" if ready else "⚪"
        state = "Available" if ready else "Not yet available"
        st.sidebar.markdown(
            f"{dot} &nbsp; <span class='phase-name'>{html.escape(label)}</span>"
            f"<br><small style='margin-left:1.4rem'>{state}</small>",
            unsafe_allow_html=True,
        )
    st.sidebar.markdown("---")


def data_status_sidebar() -> None:
    s = artifact_status()
    st.sidebar.markdown("**Live artifact status**")
    obs = f'{s["observation_count"]:,}' if s["observation_count"] is not None else "—"
    pairs = f'{s["forecasting_count"]:,}' if s["forecasting_count"] is not None else "—"
    st.sidebar.caption(f"Clean observations: {obs}")
    st.sidebar.caption(f"Forecasting pairs: {pairs}")
    st.sidebar.caption(f"Saved models: {s['model_count']}")
    st.sidebar.caption("Region: Tamil Nadu, India")
    st.sidebar.caption("Target: next observation within 150 days")


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
    border_color: str = "#26364D",
) -> None:
    st.markdown(
        f"""
        <div class="gw-card" style="border-top:2px solid {html.escape(border_color)}">
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
    root = project_root()
    model_dir = root / "models" / "trained"
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


def styled_dataframe(df, height: int = 400, use_container_width: bool = True) -> None:
    st.dataframe(df, height=height, use_container_width=use_container_width)
