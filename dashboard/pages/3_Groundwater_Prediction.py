from __future__ import annotations

import sys
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.components.ui import setup_page, page_header

DATA_PATH = ROOT / "data" / "processed" / "observations_model_ready.csv"
MODEL_DIR = ROOT / "models" / "trained"

FEATURES = [
    "GWL_m_bgl",
    "Previous_GWL_m_bgl",
    "Two_Observations_Back_GWL_m_bgl",
    "Historical_Mean_GWL_m_bgl",
    "Historical_Std_GWL_m_bgl",
    "History_Count",
    "Days_Since_Previous_Observation",
    "Current_Month",
    "Month_Sin",
    "Month_Cos",
    "Latitude_num",
    "Longitude_num",
]

MODEL_FILES = {
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
    "LightGBM": "lightgbm.joblib",
}


@st.cache_data(show_spinner="Loading cleaned CGWB observations...")
def load_observations(path_string: str) -> pd.DataFrame:
    df = pd.read_csv(path_string, low_memory=False)
    df["Parsed_Date"] = pd.to_datetime(df["Parsed_Date"], errors="coerce")
    df["GWL_m_bgl"] = pd.to_numeric(df["GWL_m_bgl"], errors="coerce")
    df = df.dropna(
        subset=["Provisional_Station_ID", "Parsed_Date", "GWL_m_bgl"]
    ).copy()
    df["Provisional_Station_ID"] = df["Provisional_Station_ID"].astype(str)

    for col, fallback in [
        ("Latitude_num", "Latitude"),
        ("Longitude_num", "Longitude"),
    ]:
        if col not in df.columns:
            df[col] = (
                pd.to_numeric(df[fallback], errors="coerce")
                if fallback in df.columns else np.nan
            )
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(
        ["Provisional_Station_ID", "Parsed_Date"]
    ).reset_index(drop=True)


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


def build_features(history: pd.DataFrame):
    """Reproduce the feature calculations used by the forecasting training script."""
    history = history.sort_values("Parsed_Date").copy()
    levels = history["GWL_m_bgl"].astype(float)
    current = history.iloc[-1]

    previous = float(levels.iloc[-2]) if len(levels) >= 2 else np.nan
    two_back = float(levels.iloc[-3]) if len(levels) >= 3 else np.nan
    previous_gap = (
        (current["Parsed_Date"] - history.iloc[-2]["Parsed_Date"]).total_seconds()
        / 86400 if len(history) >= 2 else np.nan
    )

    month = int(current["Parsed_Date"].month)
    features = {
        "GWL_m_bgl": float(current["GWL_m_bgl"]),
        "Previous_GWL_m_bgl": previous,
        "Two_Observations_Back_GWL_m_bgl": two_back,
        "Historical_Mean_GWL_m_bgl": float(levels.mean()),
        "Historical_Std_GWL_m_bgl": (
            float(levels.std(ddof=1)) if len(history) >= 2 else np.nan
        ),
        "History_Count": int(len(history)),
        "Days_Since_Previous_Observation": previous_gap,
        "Current_Month": month,
        "Month_Sin": math.sin(2 * math.pi * month / 12),
        "Month_Cos": math.cos(2 * math.pi * month / 12),
        "Latitude_num": pd.to_numeric(
            pd.Series([current["Latitude_num"]]), errors="coerce"
        ).iloc[0],
        "Longitude_num": pd.to_numeric(
            pd.Series([current["Longitude_num"]]), errors="coerce"
        ).iloc[0],
    }
    return pd.DataFrame([[features[c] for c in FEATURES]], columns=FEATURES), features


setup_page("Groundwater Prediction", "Forecasting")
page_header(
    "Groundwater Level Prediction",
    "Choose a monitoring station, review its recorded history and estimate its next observed groundwater level."
)

if not DATA_PATH.exists():
    st.error(f"Required cleaned observation file not found: {DATA_PATH}")
    st.stop()

try:
    observations = load_observations(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not load the cleaned observation data: {exc}")
    st.stop()

if observations.empty:
    st.warning("No usable observations were found in the cleaned dataset.")
    st.stop()

st.info(
    "This model estimates the next available observation represented in the training data "
    "(target gap up to 150 days). It is not a fixed 90-day forecast and does not predict "
    "an official groundwater-stress category."
)

# --- Select a station and forecast origin ---
st.subheader("1. Choose a station and observation")
stations = sorted(observations["Provisional_Station_ID"].dropna().unique().tolist())
station_id = st.selectbox(
    "Monitoring station",
    stations,
    help="These are provisional station identifiers from the cleaned research dataset.",
)

station_data = observations[
    observations["Provisional_Station_ID"] == station_id
].sort_values("Parsed_Date").reset_index(drop=True)

if station_data.empty:
    st.warning("No observations are available for this station.")
    st.stop()

date_options = station_data["Parsed_Date"].dt.strftime("%Y-%m-%d").tolist()
selected_date_text = st.selectbox(
    "Forecast from observation dated",
    date_options,
    index=len(date_options) - 1,
)
selected_date = pd.Timestamp(selected_date_text)
history = station_data[station_data["Parsed_Date"] <= selected_date].copy()
selected_row = history.iloc[-1]

district = selected_row.get("District", "Not recorded")
district = str(district) if pd.notna(district) else "Not recorded"

a, b, c, d = st.columns(4)
a.metric("History records", f"{len(history):,}")
b.metric("Latest measured depth", f'{selected_row["GWL_m_bgl"]:.2f} m bgl')
c.metric("Forecast origin", selected_date.strftime("%d %b %Y"))
d.metric("District", district)

# History chart gives context before making a prediction.
recent_history = history.tail(30).copy()
history_fig = px.line(
    recent_history,
    x="Parsed_Date",
    y="GWL_m_bgl",
    markers=True,
    title="Recent observed groundwater history",
    labels={
        "Parsed_Date": "Observation date",
        "GWL_m_bgl": "Groundwater depth (m bgl)",
    },
)
history_fig.update_layout(template="plotly_dark", height=360, margin=dict(t=55, b=15))
st.plotly_chart(history_fig, use_container_width=True)
st.caption(
    "This chart shows up to the 30 most recent recorded measurements before the "
    "forecast origin. Dates are not interpolated; gaps reflect the observation record."
)

with st.expander("View recent station observations"):
    display_cols = [
        col for col in [
            "Parsed_Date", "GWL_m_bgl", "District", "Latitude_num", "Longitude_num"
        ] if col in history.columns
    ]
    recent = history[display_cols].tail(12).copy()
    recent["Parsed_Date"] = recent["Parsed_Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        recent.sort_values("Parsed_Date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# --- Model choice and prediction ---
st.subheader("2. Choose a model")
available_models = {
    name: MODEL_DIR / filename
    for name, filename in MODEL_FILES.items()
    if (MODEL_DIR / filename).exists()
}

if not available_models:
    st.error("No supported trained model files were found in models/trained.")
    st.stop()

model_name = st.selectbox(
    "Trained model",
    list(available_models.keys()),
    help="Models are loaded from the saved training artefacts in models/trained.",
)
st.caption("The prediction uses the selected station's history up to the chosen date.")

if st.button(
    "Generate groundwater prediction",
    type="primary",
    use_container_width=True,
):
    try:
        model = load_model(str(available_models[model_name]))
        X, feature_values = build_features(history)
        prediction = float(model.predict(X[FEATURES])[0])

        st.session_state["groundwater_prediction_result"] = {
            "station": station_id,
            "date": selected_date_text,
            "model": model_name,
            "prediction": prediction,
            "current": float(selected_row["GWL_m_bgl"]),
            "features": feature_values,
        }
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

result = st.session_state.get("groundwater_prediction_result")

if (
    result
    and result["station"] == station_id
    and result["date"] == selected_date_text
):
    st.divider()
    st.subheader("3. Prediction result")

    predicted = result["prediction"]
    current = result["current"]
    delta = predicted - current

    r1, r2, r3 = st.columns(3)
    r1.metric("Estimated next level", f"{predicted:.2f} m bgl")
    r2.metric("Latest measured level", f"{current:.2f} m bgl")
    r3.metric("Change vs latest reading", f"{delta:+.2f} m")

    st.caption(
        f"Model: {result['model']} · Station: {result['station']} · "
        f"Forecast origin: {result['date']}"
    )

    comparison = pd.DataFrame({
        "Approach": ["Persistence baseline", result["model"]],
        "Groundwater depth (m bgl)": [current, predicted],
    })
    comparison_fig = px.bar(
        comparison,
        x="Approach",
        y="Groundwater depth (m bgl)",
        color="Approach",
        text_auto=".2f",
        title="Model estimate compared with persistence baseline",
    )
    comparison_fig.update_layout(
        template="plotly_dark",
        height=350,
        showlegend=False,
        margin=dict(t=55, b=15),
    )
    st.plotly_chart(comparison_fig, use_container_width=True)

    st.caption(
        "The persistence baseline assumes the next observed groundwater depth will "
        "equal the latest measured depth. A model prediction should be compared with "
        "this simple baseline, not interpreted in isolation."
    )

    with st.expander("Inspect the 12 model input features"):
        feature_table = pd.DataFrame({
            "Feature": FEATURES,
            "Value used": [
                result["features"].get(name, np.nan) for name in FEATURES
            ],
        })
        st.dataframe(feature_table, hide_index=True, use_container_width=True)

    st.warning(
        "This is a model estimate, not a measured value. In the project's overall "
        "temporal test, the persistence baseline outperformed the initial machine-learning "
        "models. Performance also varies by district and forecast gap. Do not use this "
        "estimate alone for groundwater-management decisions."
    )

st.divider()
st.subheader("How to read the result")
left, right = st.columns(2)
with left:
    st.markdown(
        """
        **Groundwater depth (m bgl)**
        - Means metres below ground level.
        - A larger positive value means the water is deeper below the local ground surface.
        - The change shown compares the estimate with the latest recorded measurement.
        """
    )
with right:
    st.markdown(
        """
        **Important limitations**
        - The target is the next available observation within a maximum 150-day gap.
        - Missing inputs and coordinates are handled by the saved model pipeline's imputer.
        - A depth prediction is not a groundwater-stress classification.
        - Provisional station IDs are not verified permanent well identities.
        """
    )
st.caption("CGWB Tamil Nadu groundwater research · Prediction uses saved project models")
