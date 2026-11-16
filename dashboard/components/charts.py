from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from dashboard.components.styles import PLOTLY_TEMPLATE

# Consistent chart theme: deep charcoal + violet + cyan + emerald.
pio.templates["gw_dark"] = go.layout.Template(PLOTLY_TEMPLATE)
_T = "gw_dark"

_PALETTE = [
    "#A78BFA",  # violet
    "#38BDF8",  # sky
    "#34D399",  # emerald
    "#FBBF24",  # amber
    "#FB7185",  # rose
    "#818CF8",  # indigo
]
_MUTED = "#A7B0C0"
_GRID = "rgba(148, 163, 184, 0.13)"
_PANEL = "#111827"
_ACCENT = "#A78BFA"


def _col(df, choices):
    return next((c for c in choices if c in df.columns), None)


def _polish(fig: go.Figure, height: int = 380, title: Optional[str] = None) -> go.Figure:
    """Apply consistent typography, spacing, axes and hover styling."""
    fig.update_layout(
        template=_T,
        height=height,
        title=dict(
            text=title if title is not None else fig.layout.title.text,
            x=0.015,
            xanchor="left",
            y=0.97,
            yanchor="top",
            font=dict(size=18, color="#F8FAFC", family="Inter, Segoe UI, sans-serif"),
        ),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#E5E7EB"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=24, r=24, t=66, b=38),
        hoverlabel=dict(
            bgcolor="#1F2937",
            bordercolor="#8B5CF6",
            font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#F9FAFB"),
        ),
        legend=dict(
            bgcolor="rgba(17,24,39,0.78)",
            bordercolor="rgba(148,163,184,0.16)",
            borderwidth=1,
            font=dict(size=11, color="#D1D5DB"),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=_GRID,
        zeroline=False,
        showline=False,
        tickfont=dict(size=11, color=_MUTED),
        title_font=dict(size=12, color="#D1D5DB"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=_GRID,
        zeroline=False,
        showline=False,
        tickfont=dict(size=11, color=_MUTED),
        title_font=dict(size=12, color="#D1D5DB"),
    )
    return fig


def _empty(title, message="No suitable data available"):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=13, color=_MUTED, family="Inter, Segoe UI, sans-serif"),
        align="center",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _polish(fig, height=340, title=title)


def make_india_well_map(
    df: pd.DataFrame,
    color_col: str = "GWL_m_bgl",
    size_col: Optional[str] = None,
    hover_cols: Optional[list[str]] = None,
    title: str = "Tamil Nadu Groundwater Monitoring Stations",
) -> go.Figure:
    lat = _col(df, ["Latitude_num", "latitude", "Latitude"])
    lon = _col(df, ["Longitude_num", "longitude", "Longitude"])
    value = color_col if color_col in df.columns else _col(
        df, ["GWL_m_bgl", "mean_depth_mbgl"]
    )

    if not lat or not lon:
        return _empty(title, "Coordinate columns are unavailable")

    data = df.copy()
    data[lat] = pd.to_numeric(data[lat], errors="coerce")
    data[lon] = pd.to_numeric(data[lon], errors="coerce")
    data = data.dropna(subset=[lat, lon])
    data = data[
        data[lat].between(8.0, 13.6)
        & data[lon].between(76.0, 80.5)
    ]

    if data.empty:
        return _empty(title, "No valid Tamil Nadu station coordinates")

    station = _col(data, ["Provisional_Station_ID", "well_id", "Station"])
    date = _col(data, ["Parsed_Date", "date"])

    if station and date:
        data[date] = pd.to_datetime(data[date], errors="coerce")
        data = data.sort_values(date).drop_duplicates(station, keep="last")

    hover = hover_cols or [
        "Station", "District", "Provisional_Station_ID",
        "Parsed_Date", "GWL_m_bgl"
    ]
    hover = [c for c in hover if c in data.columns]

    fig = px.scatter_map(
        data,
        lat=lat,
        lon=lon,
        color=value if value else None,
        size=size_col if size_col and size_col in data.columns else None,
        hover_name=station if station else None,
        hover_data=hover,
        color_continuous_scale=["#38BDF8", "#A78BFA", "#FBBF24", "#FB7185"],
        zoom=5.7,
        center={"lat": 10.9, "lon": 78.4},
        map_style="carto-darkmatter",
        title=title,
    )
    fig.update_traces(
        marker=dict(size=9, opacity=0.88, allowoverlap=True),
        selector=dict(type="scattermap"),
    )
    fig.update_layout(
        template=_T,
        height=620,
        margin=dict(t=62, b=0, l=0, r=0),
        title=dict(x=0.015, xanchor="left", font=dict(size=18, color="#F8FAFC")),
        coloraxis_colorbar=dict(
            title=dict(text="Depth (m bgl)", font=dict(color="#E5E7EB", size=11)),
            tickfont=dict(color=_MUTED),
            bgcolor="rgba(17,24,39,0.85)",
            outlinewidth=0,
        ),
        font=dict(family="Inter, Segoe UI, sans-serif", color="#E5E7EB"),
    )
    return fig


def make_state_bar(
    df: pd.DataFrame,
    x: str = "District",
    y: str = "GWL_m_bgl",
    title: str = "District Comparison",
) -> go.Figure:
    if x not in df or y not in df:
        return _empty(title)

    data = df.copy()
    data[y] = pd.to_numeric(data[y], errors="coerce")
    data = data.dropna(subset=[x, y])

    if data.empty:
        return _empty(title)

    data = data.sort_values(y)
    fig = px.bar(
        data,
        x=y,
        y=x,
        orientation="h",
        color=y,
        color_continuous_scale=["#38BDF8", "#A78BFA", "#FBBF24"],
        template=_T,
        title=title,
        labels={y: "Groundwater depth (m bgl)", x: ""},
        text_auto=".2f",
    )
    fig.update_traces(
        marker_line_width=0,
        textfont=dict(color="#F8FAFC", size=10),
        hovertemplate="<b>%{y}</b><br>Depth: %{x:.2f} m bgl<extra></extra>",
    )
    fig.update_layout(height=max(360, len(data) * 34), coloraxis_showscale=False)
    return _polish(fig, height=max(360, len(data) * 34), title=title)


def make_trend_chart(
    df: pd.DataFrame,
    x_col: str = "Parsed_Date",
    y_col: str = "GWL_m_bgl",
    color_col: Optional[str] = None,
    title: str = "Groundwater Level History",
    invert_y: bool = False,
) -> go.Figure:
    x_col = x_col if x_col in df else _col(df, ["Parsed_Date", "date", "Date"])
    y_col = y_col if y_col in df else _col(df, ["GWL_m_bgl", "water_level_mbgl"])

    if not x_col or not y_col:
        return _empty(title)

    data = df.copy()
    data[x_col] = pd.to_datetime(data[x_col], errors="coerce")
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna(subset=[x_col, y_col]).sort_values(x_col)

    if data.empty:
        return _empty(title)

    fig = px.line(
        data,
        x=x_col,
        y=y_col,
        color=color_col if color_col and color_col in data.columns else None,
        markers=True,
        template=_T,
        title=title,
        color_discrete_sequence=_PALETTE,
        labels={
            x_col: "Observation date",
            y_col: "Groundwater depth (m bgl)",
        },
    )
    fig.update_traces(
        line=dict(width=2.8, shape="linear"),
        marker=dict(size=5, line=dict(width=1, color="#0B1020")),
        hovertemplate="%{x|%d %b %Y}<br><b>%{y:.2f} m bgl</b><extra></extra>",
    )
    fig.update_layout(hovermode="x unified")
    if invert_y:
        fig.update_yaxes(autorange="reversed")
    return _polish(fig, height=410, title=title)


def make_model_comparison_bar(
    df: pd.DataFrame,
    metric: str = "RMSE",
    title: str = "Model Benchmark Comparison",
) -> go.Figure:
    model = _col(df, ["Model", "model"])
    split = _col(df, ["Split", "split"])
    metric_col = metric if metric in df.columns else metric.upper()

    if not model or metric_col not in df:
        return _empty(title)

    fig = px.bar(
        df,
        x=model,
        y=metric_col,
        color=split if split else model,
        barmode="group",
        template=_T,
        title=title,
        color_discrete_sequence=_PALETTE,
        labels={model: "Model", metric_col: f"{metric_col} (m)"},
        text_auto=".2f",
    )
    fig.update_traces(
        marker_line_width=0,
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:.3f}<extra></extra>",
    )
    fig.update_layout(height=420, xaxis_tickangle=-15)
    return _polish(fig, height=420, title=title)


def make_metric_gauge(
    value: float,
    reference: float,
    title: str = "RMSE",
) -> go.Figure:
    maximum = max(float(reference) * 2, float(value) * 1.2, 1.0)

    fig = go.Figure(go.Indicator(
        mode="number+delta+gauge",
        value=float(value),
        number={"valueformat": ".3f", "font": {"size": 34, "color": "#F8FAFC"}},
        delta={
            "reference": float(reference),
            "valueformat": ".3f",
            "font": {"size": 12},
            "increasing": {"color": "#FB7185"},
            "decreasing": {"color": "#34D399"},
        },
        title={"text": title, "font": {"size": 15, "color": "#D1D5DB"}},
        gauge={
            "axis": {"range": [0, maximum], "tickcolor": _MUTED},
            "bar": {"color": "#A78BFA", "thickness": 0.24},
            "bgcolor": "#1F2937",
            "bordercolor": "#374151",
            "borderwidth": 1,
            "steps": [
                {"range": [0, maximum * 0.5], "color": "rgba(52,211,153,0.12)"},
                {"range": [maximum * 0.5, maximum], "color": "rgba(167,139,250,0.10)"},
            ],
            "threshold": {
                "line": {"color": "#FBBF24", "width": 3},
                "thickness": 0.8,
                "value": float(reference),
            },
        },
    ))
    return _polish(fig, height=270, title="")


def make_shap_importance_bar(
    df: pd.DataFrame,
    feature_col: str = "Feature",
    value_col: str = "Mean_Absolute_SHAP",
    title: str = "Global SHAP Feature Importance",
) -> go.Figure:
    feature_col = feature_col if feature_col in df else _col(df, ["Feature", "feature"])
    value_col = value_col if value_col in df else _col(
        df, ["Mean_Absolute_SHAP", "mean_shap", "Importance_Percent"]
    )

    if not feature_col or not value_col:
        return _empty(title)

    data = df.copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=[feature_col, value_col]).sort_values(value_col)

    if data.empty:
        return _empty(title)

    fig = go.Figure(go.Bar(
        x=data[value_col],
        y=data[feature_col],
        orientation="h",
        marker=dict(
            color=data[value_col],
            colorscale=[[0, "#38BDF8"], [0.55, "#8B5CF6"], [1, "#34D399"]],
            line=dict(width=0),
        ),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        height=max(390, len(data) * 29),
        xaxis_title=value_col,
        yaxis_title="",
    )
    return _polish(fig, height=max(390, len(data) * 29), title=title)


def make_shap_waterfall(
    df: pd.DataFrame,
    feature_col: str = "feature",
    value_col: str = "shap_value",
    base_value: float = 0,
    title: str = "SHAP Waterfall — Single Prediction",
) -> go.Figure:
    if feature_col not in df or value_col not in df:
        return _empty(
            title,
            "Local SHAP contributions are not available. "
            "Global feature importance cannot be used as a local waterfall.",
        )

    data = df[[feature_col, value_col]].dropna().copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=[value_col])

    if data.empty:
        return _empty(title)

    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=["relative"] * len(data) + ["total"],
        y=list(data[feature_col]) + ["Prediction"],
        x=list(data[value_col]) + [0],
        base=base_value,
        increasing={"marker": {"color": "#34D399"}},
        decreasing={"marker": {"color": "#FB7185"}},
        totals={"marker": {"color": "#A78BFA"}},
        connector={"line": {"color": "#64748B", "width": 1}},
    ))
    return _polish(fig, height=450, title=title)


def make_missing_values_chart(
    df: pd.DataFrame,
    title: str = "Missing Values by Column",
) -> go.Figure:
    missing = (df.isna().mean() * 100).sort_values(ascending=True)

    fig = px.bar(
        x=missing.values,
        y=missing.index,
        orientation="h",
        template=_T,
        title=title,
        labels={"x": "Missing values (%)", "y": "Column"},
        color=missing.values,
        color_continuous_scale=["#34D399", "#FBBF24", "#FB7185"],
        text_auto=".1f",
    )
    fig.update_traces(
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>Missing: %{x:.2f}%<extra></extra>",
    )
    fig.update_layout(
        height=max(360, len(missing) * 27),
        coloraxis_showscale=False,
    )
    return _polish(fig, height=max(360, len(missing) * 27), title=title)


def make_distribution_chart(
    series: pd.Series,
    title: str = "",
) -> go.Figure:
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return _empty(title or str(series.name))

    fig = px.histogram(
        x=values,
        nbins=40,
        marginal="box",
        template=_T,
        title=title or str(series.name),
        labels={"x": series.name or "Value"},
        color_discrete_sequence=["#8B5CF6"],
        opacity=0.9,
    )
    fig.update_traces(
        marker_line_color="#C4B5FD",
        marker_line_width=0.6,
        hovertemplate="Value: %{x:.3f}<br>Count: %{y}<extra></extra>",
    )
    fig.update_layout(height=390, showlegend=False, bargap=0.06)
    return _polish(fig, height=390, title=title or str(series.name))


def make_seasonal_boxplot(
    df: pd.DataFrame,
    y_col: str = "GWL_m_bgl",
) -> go.Figure:
    if "season" not in df.columns:
        return _empty(
            "Seasonal Distribution",
            "No documented season classification is available in this dataset.",
        )

    if y_col not in df.columns:
        y_col = _col(df, ["GWL_m_bgl", "water_level_mbgl"])

    if not y_col:
        return _empty("Seasonal Distribution")

    fig = px.box(
        df,
        x="season",
        y=y_col,
        points="outliers",
        template=_T,
        title="Groundwater Depth by Supplied Season",
        labels={y_col: "Depth (m bgl)", "season": "Season"},
        color="season",
        color_discrete_sequence=_PALETTE,
    )
    fig.update_traces(
        marker=dict(size=4, color="#C4B5FD"),
        line=dict(width=2),
        fillcolor="rgba(139,92,246,0.22)",
    )
    fig.update_layout(height=400, showlegend=False)
    return _polish(fig, height=400, title="Groundwater Depth by Supplied Season")
