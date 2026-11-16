"""
dashboard/components
====================
Reusable components for the Groundwater Stress ML Dashboard.
"""

from dashboard.components.styles import (
    PLOTLY_TEMPLATE,
    inject_theme,
    demo_banner,
    section_header,
    status_pill,
    insight_card,
)
from dashboard.components.ui import (
    setup_page,
    page_header,
    sidebar_branding,
    sidebar_pipeline_status,
    data_status_sidebar,
    metric_row,
    kpi_card,
    no_data_state,
    model_unavailable_state,
    pending_badge,
    success_badge,
    demo_badge,
    project_root,
    styled_dataframe,
)

__all__ = [
    "PLOTLY_TEMPLATE",
    "inject_theme",
    "demo_banner",
    "section_header",
    "status_pill",
    "insight_card",
    "setup_page",
    "page_header",
    "sidebar_branding",
    "sidebar_pipeline_status",
    "data_status_sidebar",
    "metric_row",
    "kpi_card",
    "no_data_state",
    "model_unavailable_state",
    "pending_badge",
    "success_badge",
    "demo_badge",
    "project_root",
    "styled_dataframe",
]
