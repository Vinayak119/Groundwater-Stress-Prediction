from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.components.ui import setup_page, page_header
from dashboard.components.styles import section_header
from dashboard.components.charts import make_india_well_map, make_trend_chart

root = setup_page(title="Geographic Analysis", icon="🗺️")

page_header(
    "🗺️",
    "Tamil Nadu Geographic Analysis",
    "Actual CGWB groundwater observations, station coverage and district-level validation."
)

DATA_PATH = root / "data" / "processed" / "observations_model_ready.csv"
GEO_PATH = root / "results" / "metrics" / "geographic_holdout_results.csv"
DISTRICT_PATH = root / "results" / "metrics" / "geographic_district_performance.csv"


@st.cache_data
def load_csv(path_string):
    path = Path(path_string)
    if not path.exists():
        return None
    return pd.read_csv(path)


df = load_csv(str(DATA_PATH))

if df is None:
    st.error("The cleaned CGWB observation file was not found.")
    st.caption("Expected: data/processed/observations_model_ready.csv")
    st.stop()

required = ["Parsed_Date", "GWL_m_bgl"]
missing = [column for column in required if column not in df.columns]

if missing:
    st.error(f"Required observation columns are missing: {missing}")
    st.stop()

df["Parsed_Date"] = pd.to_datetime(df["Parsed_Date"], errors="coerce")
df["GWL_m_bgl"] = pd.to_numeric(df["GWL_m_bgl"], errors="coerce")
df = df.dropna(subset=["Parsed_Date", "GWL_m_bgl"]).copy()

station_col = next(
    (c for c in ["Provisional_Station_ID", "Station_ID", "Station"] if c in df.columns),
    None
)
district_col = next(
    (c for c in ["District", "District_Name", "district"] if c in df.columns),
    None
)
lat_col = next(
    (c for c in ["Latitude_num", "Latitude", "latitude"] if c in df.columns),
    None
)
lon_col = next(
    (c for c in ["Longitude_num", "Longitude", "longitude"] if c in df.columns),
    None
)

st.info(
    "Research data mode: all measurements shown on this page come from the "
    "cleaned CGWB observation file. No synthetic station records are used."
)

section_header("Observation coverage")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Clean observations", f"{len(df):,}")
c2.metric(
    "Provisional stations",
    f"{df[station_col].nunique():,}" if station_col else "Unavailable"
)
c3.metric(
    "Districts represented",
    f"{df[district_col].nunique():,}" if district_col else "Unavailable"
)
c4.metric(
    "Stations with coordinates",
    f"{df.dropna(subset=[lat_col, lon_col])[station_col].nunique():,}"
    if station_col and lat_col and lon_col else "Unavailable"
)

st.caption(
    f"Observation dates: {df['Parsed_Date'].min():%d %b %Y} to "
    f"{df['Parsed_Date'].max():%d %b %Y}. "
    "Station identifiers are provisional and are not verified official well identities."
)

section_header("Station map")

if lat_col and lon_col:
    map_df = df.copy()
    map_df[lat_col] = pd.to_numeric(map_df[lat_col], errors="coerce")
    map_df[lon_col] = pd.to_numeric(map_df[lon_col], errors="coerce")
    map_df = map_df.dropna(subset=[lat_col, lon_col])

    if district_col:
        districts = sorted(map_df[district_col].dropna().astype(str).unique())
        selected_districts = st.multiselect(
            "Filter map by district",
            districts,
            default=districts
        )
        map_df = map_df[
            map_df[district_col].astype(str).isin(selected_districts)
        ]

    if not map_df.empty:
        fig = make_india_well_map(
            map_df,
            color_col="GWL_m_bgl",
            title="Latest available groundwater observation by station"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "The map uses the latest available observation for each provisional "
            "station with usable coordinates. Marker colour represents measured "
            "depth in metres below ground level, not an official stress category."
        )
    else:
        st.warning("No coordinate-bearing records match the selected districts.")
else:
    st.warning(
        "Latitude and longitude columns were not found. The map is unavailable, "
        "but station time-series analysis remains available."
    )

section_header("Station-level observation history")

if station_col:
    station_ids = sorted(df[station_col].dropna().astype(str).unique().tolist())

    if station_ids:
        selected_station = st.selectbox(
            "Choose a provisional station",
            station_ids
        )

        station_df = df[
            df[station_col].astype(str) == selected_station
        ].sort_values("Parsed_Date")

        a, b, c = st.columns(3)
        a.metric("Observations", f"{len(station_df):,}")
        b.metric(
            "First observation",
            station_df["Parsed_Date"].min().strftime("%d %b %Y")
        )
        c.metric(
            "Latest observation",
            station_df["Parsed_Date"].max().strftime("%d %b %Y")
        )

        if district_col and station_df[district_col].notna().any():
            st.caption(
                "District: " +
                str(station_df[district_col].dropna().iloc[0])
            )

        fig = make_trend_chart(
            station_df,
            x_col="Parsed_Date",
            y_col="GWL_m_bgl",
            title=f"Observed groundwater depth — {selected_station}",
            invert_y=False
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "This is the recorded observation history. Irregular dates are retained; "
            "the line does not imply that measurements were taken at equal intervals."
        )

        with st.expander("View station observations"):
            st.dataframe(
                station_df.sort_values("Parsed_Date", ascending=False),
                use_container_width=True,
                hide_index=True
            )
else:
    st.warning("A provisional station identifier column was not found.")

section_header("Geographic holdout experiment")

geo = load_csv(str(GEO_PATH))
district_results = load_csv(str(DISTRICT_PATH))

if geo is not None:
    st.subheader("Overall geographic holdout metrics")
    st.dataframe(geo, use_container_width=True, hide_index=True)
    st.caption(
        "This experiment holds out seven Tamil Nadu districts. "
        "It is not a cross-state or independent external validation."
    )
else:
    st.warning("Geographic holdout summary is unavailable.")

if district_results is not None:
    st.subheader("District-level performance")
    st.dataframe(
        district_results,
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("District-level performance file is unavailable.")

section_header("Interpretation and limitations")

st.markdown("""
- The displayed variable is groundwater depth in metres below ground level (m bgl).
- Greater depth is not, by itself, proof of groundwater stress.
- Official stress classification requires an appropriate hydrogeological
  definition and supporting evidence.
- Missing coordinates prevent some observations from appearing on the map;
  they do not make those observations unusable for time-series analysis.
- Provisional station IDs may not establish permanent well identity.
- Anomalous readings are retained in the research data pending verification.
- The geographic experiment evaluates held-out districts within Tamil Nadu,
  not transfer to another state.
""")

st.caption("CGWB Tamil Nadu research data · Geographic analysis")
