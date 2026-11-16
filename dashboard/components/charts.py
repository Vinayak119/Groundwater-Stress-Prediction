from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from dashboard.components.styles import PLOTLY_TEMPLATE

pio.templates["gw_dark"] = go.layout.Template(PLOTLY_TEMPLATE)
_T = "gw_dark"


def _col(df, choices):
    return next((c for c in choices if c in df.columns), None)


def _empty(title, message="No suitable data available"):
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=.5, y=.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=14, color="#91A5BF")
    )
    fig.update_layout(template=_T, title=title, height=340)
    return fig


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
        data[lat].between(8.0, 13.6) &
        data[lon].between(76.0, 80.5)
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
        hover_name=station if station else None,
        hover_data=hover,
        color_continuous_scale=["#36C5D9", "#F5B95F", "#E87979"],
        zoom=5.7,
        center={"lat": 10.9, "lon": 78.4},
        map_style="carto-darkmatter",
        title=title,
    )
    fig.update_traces(marker=dict(size=9, opacity=.82))
    fig.update_layout(
        template=_T, height=600,
        margin=dict(t=50, b=0, l=0, r=0),
        coloraxis_colorbar_title="Depth (m bgl)"
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

    fig = px.bar(
        data.sort_values(y), x=y, y=x, orientation="h",
        color=y, color_continuous_scale=["#36C5D9", "#6E9EFF", "#F5B95F"],
        template=_T, title=title,
        labels={y: "Groundwater depth (m bgl)", x: ""}
    )
    fig.update_layout(
        height=max(340, len(data) * 32),
        coloraxis_showscale=False
    )
    return fig


def make_trend_chart(
    df: pd.DataFrame,
    x_col: str = "Parsed_Date",
    y_col: str = "GWL_m_bgl",
    color_col: Optional[str] = None,
    title: str = "Groundwater Level History",
    invert_y: bool = False,
) -> go.Figure:
    x_col = x_col if x_col in df else _col(
        df, ["Parsed_Date", "date", "Date"]
    )
    y_col = y_col if y_col in df else _col(
        df, ["GWL_m_bgl", "water_level_mbgl"]
    )

    if not x_col or not y_col:
        return _empty(title)

    data = df.copy()
    data[x_col] = pd.to_datetime(data[x_col], errors="coerce")
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna(subset=[x_col, y_col]).sort_values(x_col)

    if data.empty:
        return _empty(title)

    fig = px.line(
        data, x=x_col, y=y_col,
        color=color_col if color_col and color_col in data.columns else None,
        markers=True, template=_T, title=title,
        labels={
            x_col: "Observation date",
            y_col: "Groundwater depth (m bgl)"
        }
    )
    fig.update_traces(line=dict(width=2.4), marker=dict(size=5))
    fig.update_layout(height=390, hovermode="x unified")

    if invert_y:
        fig.update_yaxes(autorange="reversed")

    return fig


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
        df, x=model, y=metric_col,
        color=split if split else model,
        barmode="group", template=_T, title=title,
        labels={model: "Model", metric_col: f"{metric_col} (m)"}
    )
    fig.update_layout(height=390, xaxis_tickangle=-20)
    return fig


def make_metric_gauge(
    value: float,
    reference: float,
    title: str = "RMSE",
) -> go.Figure:
    maximum = max(float(reference) * 2, float(value) * 1.2, 1.0)

    fig = go.Figure(go.Indicator(
        mode="number+delta+gauge",
        value=float(value),
        delta={"reference": float(reference), "valueformat": ".3f"},
        title={"text": title},
        gauge={
            "axis": {"range": [0, maximum]},
            "bar": {"color": "#36C5D9"},
            "bgcolor": "#17243A",
            "bordercolor": "#26364D",
            "threshold": {
                "line": {"color": "#F5B95F", "width": 3},
                "thickness": .8,
                "value": float(reference)
            }
        }
    ))
    fig.update_layout(template=_T, height=240)
    return fig


def make_shap_importance_bar(
    df: pd.DataFrame,
    feature_col: str = "Feature",
    value_col: str = "Mean_Absolute_SHAP",
    title: str = "Global SHAP Feature Importance",
) -> go.Figure:
    feature_col = feature_col if feature_col in df else _col(
        df, ["Feature", "feature"]
    )
    value_col = value_col if value_col in df else _col(
        df, ["Mean_Absolute_SHAP", "mean_shap", "Importance_Percent"]
    )

    if not feature_col or not value_col:
        return _empty(title)

    data = df.copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=[feature_col, value_col]).sort_values(value_col)

    fig = go.Figure(go.Bar(
        x=data[value_col], y=data[feature_col], orientation="h",
        marker=dict(color=data[value_col], colorscale="Tealgrn"),
        hovertemplate="<b>%{y}</b><br>%{x:.4f}<extra></extra>"
    ))
    fig.update_layout(
        template=_T, title=title,
        height=max(380, len(data) * 25),
        xaxis_title=value_col, yaxis_title=""
    )
    return fig


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
            "Global feature importance cannot be used as a local waterfall."
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
        increasing={"marker": {"color": "#42D6A4"}},
        decreasing={"marker": {"color": "#E87979"}},
        totals={"marker": {"color": "#36C5D9"}}
    ))
    fig.update_layout(template=_T, title=title, height=430)
    return fig


def make_missing_values_chart(
    df: pd.DataFrame,
    title: str = "Missing Values by Column",
) -> go.Figure:
    missing = (df.isna().mean() * 100).sort_values(ascending=True)

    fig = px.bar(
        x=missing.values, y=missing.index, orientation="h",
        template=_T, title=title,
        labels={"x": "Missing values (%)", "y": "Column"},
        color=missing.values,
        color_continuous_scale=["#42D6A4", "#F5B95F", "#E87979"]
    )
    fig.update_layout(
        height=max(340, len(missing) * 24),
        coloraxis_showscale=False
    )
    return fig


def make_distribution_chart(
    series: pd.Series,
    title: str = "",
) -> go.Figure:
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return _empty(title or str(series.name))

    fig = px.histogram(
        x=values, nbins=40, marginal="box",
        template=_T, title=title or str(series.name),
        labels={"x": series.name or "Value"},
        color_discrete_sequence=["#36C5D9"]
    )
    fig.update_layout(height=350, showlegend=False)
    return fig


def make_seasonal_boxplot(
    df: pd.DataFrame,
    y_col: str = "GWL_m_bgl",
) -> go.Figure:
    if "season" not in df.columns:
        return _empty(
            "Seasonal Distribution",
            "No documented season classification is available in this dataset."
        )

    if y_col not in df.columns:
        y_col = _col(df, ["GWL_m_bgl", "water_level_mbgl"])

    if not y_col:
        return _empty("Seasonal Distribution")

    fig = px.box(
        df, x="season", y=y_col, points="outliers",
        template=_T, title="Groundwater Depth by Supplied Season",
        labels={y_col: "Depth (m bgl)", "season": "Season"}
    )
    fig.update_layout(height=370)
    return fig

