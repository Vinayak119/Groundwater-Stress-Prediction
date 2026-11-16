from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.components.ui import setup_page, page_header
from dashboard.components.styles import section_header

root = setup_page(title="Data Explorer", icon="🔎")

page_header(
    "🔎",
    "Data Explorer & Quality Audit",
    "Real CGWB Tamil Nadu observations · Data quality · Historical groundwater measurements"
)

DATA_PATH = root / "data" / "processed" / "observations_model_ready.csv"


@st.cache_data(show_spinner="Loading cleaned CGWB observations...")
def load_data(path_string):
    path = Path(path_string)
    if not path.exists():
        return None
    data = pd.read_csv(path)
    return data


df_full = load_data(str(DATA_PATH))

if df_full is None:
    st.error("Cleaned CGWB observation data was not found.")
    st.caption("Expected: data/processed/observations_model_ready.csv")
    st.stop()

required = ["Parsed_Date", "GWL_m_bgl"]
missing = [c for c in required if c not in df_full.columns]

if missing:
    st.error(f"Required columns are missing: {missing}")
    st.stop()

df_full["Parsed_Date"] = pd.to_datetime(
    df_full["Parsed_Date"], errors="coerce"
)
df_full["GWL_m_bgl"] = pd.to_numeric(
    df_full["GWL_m_bgl"], errors="coerce"
)

df_full = df_full.dropna(
    subset=["Parsed_Date", "GWL_m_bgl"]
).copy()

station_col = next(
    (c for c in ["Provisional_Station_ID", "Station_ID", "Station"] if c in df_full.columns),
    None
)
district_col = next(
    (c for c in ["District", "District_Name", "district"] if c in df_full.columns),
    None
)

df_full["Observation_Year"] = df_full["Parsed_Date"].dt.year
df_full["Observation_Month"] = df_full["Parsed_Date"].dt.month

st.success(
    f"Real CGWB research data loaded: {len(df_full):,} cleaned observations. "
    "No synthetic demo records are used."
)

st.caption(
    "Source: data/processed/observations_model_ready.csv. "
    "The displayed GWL variable is groundwater depth in metres below ground level (m bgl)."
)

section_header("Dataset overview")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Clean observations", f"{len(df_full):,}")
c2.metric(
    "Provisional stations",
    f"{df_full[station_col].nunique():,}" if station_col else "Unavailable"
)
c3.metric(
    "Districts",
    f"{df_full[district_col].nunique():,}" if district_col else "Unavailable"
)
c4.metric(
    "Mean depth",
    f"{df_full['GWL_m_bgl'].mean():.2f} m"
)
c5.metric(
    "Missing coordinates",
    f"{df_full[['Latitude_num', 'Longitude_num']].isna().any(axis=1).sum():,}"
    if {"Latitude_num", "Longitude_num"}.issubset(df_full.columns)
    else "Not assessed"
)

st.caption(
    f"Observed date range: {df_full['Parsed_Date'].min():%d %b %Y} "
    f"to {df_full['Parsed_Date'].max():%d %b %Y}."
)

section_header("Filter observations")

f1, f2, f3 = st.columns(3)

with f1:
    years = sorted(df_full["Observation_Year"].dropna().unique().tolist())
    selected_years = st.multiselect(
        "Observation years",
        years,
        default=years,
        key="explorer_years"
    )

with f2:
    if district_col:
        districts = sorted(
            df_full[district_col].dropna().astype(str).unique().tolist()
        )
        selected_districts = st.multiselect(
            "Districts",
            districts,
            default=districts,
            key="explorer_districts"
        )
    else:
        selected_districts = []

with f3:
    search = st.text_input(
        "Search station ID",
        key="explorer_station_search"
    )

df = df_full.copy()

if selected_years:
    df = df[df["Observation_Year"].isin(selected_years)]
else:
    df = df.iloc[0:0]

if district_col and selected_districts:
    df = df[df[district_col].astype(str).isin(selected_districts)]
elif district_col and not selected_districts:
    df = df.iloc[0:0]

if search and station_col:
    df = df[
        df[station_col].astype(str).str.contains(
            search, case=False, na=False
        )
    ]

st.caption(f"Current filter result: {len(df):,} observations")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Record Preview",
    "Depth Distribution",
    "Data Quality",
    "Observation Trends",
    "Research Notes"
])

with tab1:
    section_header("Actual CGWB observation records")

    if not df.empty:
        sort_options = [
            c for c in ["Parsed_Date", "GWL_m_bgl", station_col, district_col]
            if c and c in df.columns
        ]

        sort_col = st.selectbox(
            "Sort records by",
            sort_options,
            index=0
        )

        ascending = st.checkbox("Oldest / lowest first", value=False)

        display_df = df.sort_values(
            sort_col, ascending=ascending
        )

        st.dataframe(
            display_df.head(1000),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download all filtered observations (CSV)",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="cgwb_filtered_observations.csv",
            mime="text/csv"
        )

        st.caption(
            f"Showing up to 1,000 rows on screen; CSV download contains "
            f"all {len(display_df):,} filtered records."
        )
    else:
        st.warning("No records match the current filters.")

with tab2:
    section_header("Groundwater depth distribution")

    if not df.empty:
        fig = px.histogram(
            df,
            x="GWL_m_bgl",
            nbins=60,
            title="Distribution of observed groundwater depth",
            labels={"GWL_m_bgl": "Groundwater depth (m bgl)", "count": "Records"}
        )
        fig.update_layout(template="plotly_dark", height=430)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Descriptive statistics")

        stats = df["GWL_m_bgl"].describe().to_frame(
            name="Groundwater depth (m bgl)"
        )
        st.dataframe(stats, use_container_width=True)

        negative_count = int((df["GWL_m_bgl"] < 0).sum())
        st.info(
            f"{negative_count:,} negative groundwater readings are present "
            "in the filtered data. They are retained and are not automatically "
            "treated as errors."
        )
    else:
        st.warning("No observations are available for this distribution.")

with tab3:
    section_header("Missing values and data types")

    quality = pd.DataFrame({
        "Column": df_full.columns,
        "Data type": df_full.dtypes.astype(str).values,
        "Missing records": df_full.isna().sum().values,
        "Missing (%)": (df_full.isna().mean() * 100).round(2).values,
        "Unique values": df_full.nunique(dropna=True).values
    })

    st.dataframe(
        quality.sort_values("Missing (%)", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    missing_plot = quality[quality["Missing records"] > 0].sort_values(
        "Missing (%)", ascending=True
    )

    if not missing_plot.empty:
        fig = px.bar(
            missing_plot,
            x="Missing (%)",
            y="Column",
            orientation="h",
            title="Missing values by column"
        )
        fig.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values found in the loaded cleaned dataset.")

    st.caption(
        "Missingness is assessed on the cleaned observation file, not the original raw files."
    )

with tab4:
    section_header("Groundwater observation history")

    if station_col and not df.empty:
        station_ids = sorted(
            df[station_col].dropna().astype(str).unique().tolist()
        )

        selected_station = st.selectbox(
            "Select a provisional station",
            station_ids,
            key="explorer_history_station"
        )

        station_df = df[
            df[station_col].astype(str) == selected_station
        ].sort_values("Parsed_Date")

        fig = px.line(
            station_df,
            x="Parsed_Date",
            y="GWL_m_bgl",
            markers=True,
            title=f"Observed groundwater depth: {selected_station}",
            labels={
                "Parsed_Date": "Observation date",
                "GWL_m_bgl": "Groundwater depth (m bgl)"
            }
        )
        fig.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Measurements are plotted at their recorded dates. "
            "Observation intervals are irregular and are not interpolated."
        )

        if len(station_df) > 1:
            st.metric("Station observations", f"{len(station_df):,}")
            st.metric(
                "Time between first and last observation",
                f"{(station_df['Parsed_Date'].max() - station_df['Parsed_Date'].min()).days:,} days"
            )
    else:
        st.warning("Station history is unavailable for the current filters.")

    section_header("Monthly observation counts")

    monthly = (
        df.groupby(df["Parsed_Date"].dt.month)
        .size()
        .reindex(range(1, 13), fill_value=0)
        .rename_axis("Month")
        .reset_index(name="Observation count")
    )

    fig = px.bar(
        monthly,
        x="Month",
        y="Observation count",
        title="Number of observations by calendar month"
    )
    fig.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "This chart describes sampling frequency by calendar month; "
        "it does not assign monsoon or recharge seasons."
    )

with tab5:
    section_header("Research interpretation and limitations")

    st.markdown("""
    **Data provenance**
    - This page uses the cleaned CGWB observation file in the project.
    - The displayed records are measurements, not synthetic examples.
    - Station identifiers are provisional and are not verified official well identities.

    **Measurement interpretation**
    - GWL is represented in metres below ground level (m bgl).
    - Groundwater depth is not the same as groundwater elevation above mean sea level.
    - Negative readings are retained pending hydrogeological verification.
    - A larger depth below ground level does not alone establish groundwater stress.

    **Data-quality handling**
    - Conflicting station/date records were quarantined during preprocessing.
    - Missing coordinates do not automatically make a measurement unusable for temporal analysis.
    - Irregular observation dates are retained rather than converted into artificial quarterly dates.
    - Unusual values require source verification; they are not automatically deleted.

    **Research scope**
    - This dataset supports Tamil Nadu groundwater-level analysis and next-observation forecasting.
    - Official stress classification requires an independently defined hydrogeological assessment method.
    """)

st.caption("CGWB Tamil Nadu groundwater research · Real-data explorer")
