from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.components.ui import setup_page, page_header

ROOT = Path(__file__).resolve().parents[2]
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


@st.cache_data
def load_observations():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["Parsed_Date"] = pd.to_datetime(df["Parsed_Date"], errors="coerce")
    df["GWL_m_bgl"] = pd.to_numeric(df["GWL_m_bgl"], errors="coerce")
    df = df.dropna(subset=["Provisional_Station_ID", "Parsed_Date", "GWL_m_bgl"])
    df["Provisional_Station_ID"] = df["Provisional_Station_ID"].astype(str)
    for col in ["Latitude_num", "Longitude_num"]:
        if col not in df.columns:
            source = "Latitude" if col == "Latitude_num" else "Longitude"
            df[col] = pd.to_numeric(df[source], errors="coerce") if source in df.columns else np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["Provisional_Station_ID", "Parsed_Date"]).reset_index(drop=True)


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


def build_features(history):
    """Reproduce the feature calculations used by the forecasting training script."""
    history = history.sort_values("Parsed_Date").copy()
    levels = history["GWL_m_bgl"].astype(float)
    current = history.iloc[-1]

    previous = float(levels.iloc[-2]) if len(levels) >= 2 else np.nan
    two_back = float(levels.iloc[-3]) if len(levels) >= 3 else np.nan
    previous_gap = (
        (current["Parsed_Date"] - history.iloc[-2]["Parsed_Date"]).total_seconds() / 86400
        if len(history) >= 2 else np.nan
    )

    month = int(current["Parsed_Date"].month)
    features = {
        "GWL_m_bgl": float(current["GWL_m_bgl"]),
        "Previous_GWL_m_bgl": previous,
        "Two_Observations_Back_GWL_m_bgl": two_back,
        "Historical_Mean_GWL_m_bgl": float(levels.mean()),
        "Historical_Std_GWL_m_bgl": float(levels.std(ddof=1)) if len(levels) >= 2 else np.nan,
        "History_Count": int(len(history)),
        "Days_Since_Previous_Observation": previous_gap,
        "Current_Month": month,
        "Month_Sin": math.sin(2 * math.pi * month / 12),
        "Month_Cos": math.cos(2 * math.pi * month / 12),
        "Latitude_num": pd.to_numeric(pd.Series([current["Latitude_num"]]), errors="coerce").iloc[0],
        "Longitude_num": pd.to_numeric(pd.Series([current["Longitude_num"]]), errors="coerce").iloc[0],
    }
    return pd.DataFrame([[features[c] for c in FEATURES]], columns=FEATURES), features


setup_page("Groundwater Prediction", "Forecasting")
page_header(
    "Groundwater Level Prediction",
    "Forecast the next observed groundwater level using the selected station's actual CGWB observation history.",
)

if not DATA_PATH.exists():
    st.error(f"Required cleaned observation file not found: {DATA_PATH}")
    st.stop()

try:
    observations = load_observations()
except Exception as exc:
    st.error(f"Could not load the cleaned observation data: {exc}")
    st.stop()

if observations.empty:
    st.warning("No usable observations were found in the cleaned dataset.")
    st.stop()

st.info(
    "Research scope: the models were trained to predict the next available observation "
    "within a maximum 150-day target gap. This is not a guaranteed 90-day forecast. "
    "Predictions are groundwater levels in metres below ground level (m bgl), not stress classes."
)

stations = sorted(observations["Provisional_Station_ID"].dropna().unique().tolist())
station_id = st.selectbox("Select provisional monitoring-station ID", stations)

station_data = observations[
    observations["Provisional_Station_ID"] == station_id
].sort_values("Parsed_Date").reset_index(drop=True)

if station_data.empty:
    st.warning("No observations are available for this station.")
    st.stop()

date_options = station_data["Parsed_Date"].dt.strftime("%Y-%m-%d").tolist()
selected_date_text = st.selectbox(
    "Choose the observation to forecast from",
    date_options,
    index=len(date_options) - 1,
)
selected_date = pd.Timestamp(selected_date_text)

history = station_data[station_data["Parsed_Date"] <= selected_date].copy()
selected_row = history.iloc[-1]

st.subheader("Selected station and observation")
a, b, c, d = st.columns(4)
a.metric("Observations in history", f"{len(history):,}")
b.metric("Current groundwater level", f'{selected_row["GWL_m_bgl"]:.2f} m bgl')
c.metric("Observation date", selected_date.strftime("%d %b %Y"))
district = selected_row.get("District", "Not recorded")
d.metric("District", str(district) if pd.notna(district) else "Not recorded")

with st.expander("View recent station observations"):
    display_cols = [
        c for c in [
            "Parsed_Date", "GWL_m_bgl", "District",
            "Latitude_num", "Longitude_num"
        ] if c in history.columns
    ]
    recent = history[display_cols].tail(12).copy()
    recent["Parsed_Date"] = recent["Parsed_Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(recent.sort_values("Parsed_Date", ascending=False), use_container_width=True)

st.divider()
st.subheader("Generate a prediction")

available_models = {
    name: MODEL_DIR / filename
    for name, filename in MODEL_FILES.items()
    if (MODEL_DIR / filename).exists()
}

if not available_models:
    st.error("No supported trained model files were found in models/trained.")
    st.stop()

model_name = st.selectbox("Select trained model", list(available_models.keys()))
st.caption("The selected model is loaded from the project's trained model artefacts.")

if st.button("Predict next observed groundwater level", type="primary", use_container_width=True):
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

if result and result["station"] == station_id and result["date"] == selected_date_text:
    st.divider()
    st.subheader("Prediction result")

    predicted = result["prediction"]
    current = result["current"]
    delta = predicted - current

    x, y, z = st.columns(3)
    x.metric("Predicted next observed level", f"{predicted:.2f} m bgl")
    y.metric("Current observed level", f"{current:.2f} m bgl")
    z.metric("Predicted change", f"{delta:+.2f} m")

    st.caption(
        f"Model: {result['model']} · Station: {result['station']} · "
        f"Forecast origin: {result['date']}"
    )

    st.markdown(
        "**Persistence baseline:** "
        f"{current:.2f} m bgl. This baseline assumes the next observed level "
        "will equal the current observed level."
    )

    comparison = pd.DataFrame({
        "Method": ["Persistence baseline", result["model"]],
        "Predicted groundwater level (m bgl)": [current, predicted],
    })
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    with st.expander("Inspect the 12 model input features"):
        feature_table = pd.DataFrame({
            "Feature": FEATURES,
            "Value used": [result["features"].get(name, np.nan) for name in FEATURES],
        })
        st.dataframe(feature_table, hide_index=True, use_container_width=True)

    st.warning(
        "This is a model estimate, not a measured value. Model performance varies by "
        "district and forecast gap. The temporal benchmark found that persistence "
        "outperformed the initial machine-learning models on the overall test set. "
        "Do not use this result alone for groundwater-management decisions."
    )

st.divider()
st.subheader("How to interpret this forecast")
st.markdown(
    """
    - **m bgl** means metres below ground level. A larger positive value means the
      water level is deeper below the local ground surface.
    - The forecast estimates the next available observation represented in the
      training data, where the target gap was at most 150 days.
    - Missing historical inputs or coordinates are handled by the imputation step
      inside the saved model pipeline.
    - A groundwater-stress category is not inferred here. Stress classification
      requires an independently justified hydrogeological or official assessment
      threshold, not a depth prediction alone.
    """
)
