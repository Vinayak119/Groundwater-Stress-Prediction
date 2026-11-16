from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/observations_model_ready.csv"
OUT = ROOT / "data/processed/forecasting_pairs_next_observation_150d.csv"
SUMMARY = ROOT / "reports/forecasting_dataset_summary.json"

d = pd.read_csv(SOURCE)
d["Parsed_Date"] = pd.to_datetime(d["Parsed_Date"], errors="coerce")
d["GWL_m_bgl"] = pd.to_numeric(d["GWL_m_bgl"], errors="coerce")

d = d.dropna(subset=[
    "Provisional_Station_ID", "Parsed_Date", "GWL_m_bgl"
]).copy()

d = d.sort_values(
    ["Provisional_Station_ID", "Parsed_Date"]
).reset_index(drop=True)

g = d.groupby("Provisional_Station_ID", sort=False)

# Information available at the current observation
d["Previous_GWL_m_bgl"] = g["GWL_m_bgl"].shift(1)
d["Two_Observations_Back_GWL_m_bgl"] = g["GWL_m_bgl"].shift(2)

d["Historical_Mean_GWL_m_bgl"] = (
    d.groupby("Provisional_Station_ID")["GWL_m_bgl"]
    .transform(lambda x: x.expanding().mean())
)

d["Historical_Std_GWL_m_bgl"] = (
    d.groupby("Provisional_Station_ID")["GWL_m_bgl"]
    .transform(lambda x: x.expanding().std())
)

d["History_Count"] = g.cumcount() + 1

previous_date = g["Parsed_Date"].shift(1)
d["Days_Since_Previous_Observation"] = (
    d["Parsed_Date"] - previous_date
).dt.total_seconds() / 86400

d["Current_Month"] = d["Parsed_Date"].dt.month
d["Month_Sin"] = np.sin(2 * np.pi * d["Current_Month"] / 12)
d["Month_Cos"] = np.cos(2 * np.pi * d["Current_Month"] / 12)

# Next observation is the prediction target, never an input feature
d["Target_Date"] = g["Parsed_Date"].shift(-1)
d["Target_GWL_m_bgl"] = g["GWL_m_bgl"].shift(-1)

d["Target_Gap_Days"] = (
    d["Target_Date"] - d["Parsed_Date"]
).dt.total_seconds() / 86400

# Keep only future observations within the 150-day study horizon
pairs = d[
    d["Target_Gap_Days"].between(0.000001, 150, inclusive="both")
].copy()

# Split using target dates, not current observation dates
pairs["Split"] = np.select(
    [
        pairs["Target_Date"] <= pd.Timestamp("2018-12-31"),
        pairs["Target_Date"].between(
            "2019-01-01", "2020-12-31"
        ),
        pairs["Target_Date"] >= pd.Timestamp("2021-01-01"),
    ],
    ["train", "validation", "test"],
    default="unassigned"
)

pairs = pairs[pairs["Split"] != "unassigned"].copy()

columns = [
    "Provisional_Station_ID", "Station", "District",
    "Latitude_num", "Longitude_num",
    "Parsed_Date", "Target_Date", "Target_Gap_Days",
    "GWL_m_bgl", "Previous_GWL_m_bgl",
    "Two_Observations_Back_GWL_m_bgl",
    "Historical_Mean_GWL_m_bgl",
    "Historical_Std_GWL_m_bgl",
    "History_Count", "Days_Since_Previous_Observation",
    "Current_Month", "Month_Sin", "Month_Cos",
    "Target_GWL_m_bgl", "Split"
]

pairs[columns].to_csv(OUT, index=False)

summary = {
    "source_observations": int(len(d)),
    "forecasting_pairs": int(len(pairs)),
    "unique_provisional_stations": int(
        pairs["Provisional_Station_ID"].nunique()
    ),
    "maximum_target_gap_days": 150,
    "target_definition": "Next available observation within 150 days",
    "split_basis": "Target date",
    "split_counts": {
        str(k): int(v)
        for k, v in pairs["Split"].value_counts().items()
    },
    "target_gap_days": {
        "median": float(pairs["Target_Gap_Days"].median()),
        "minimum": float(pairs["Target_Gap_Days"].min()),
        "maximum": float(pairs["Target_Gap_Days"].max())
    },
    "split_target_date_ranges": {
        str(name): {
            "first": str(group["Target_Date"].min()),
            "last": str(group["Target_Date"].max())
        }
        for name, group in pairs.groupby("Split")
    },
    "future_target_gap_used_as_feature": False
}

SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("FORECASTING DATASET CREATED")
print("Source observations:", len(d))
print("Forecasting pairs:", len(pairs))
print("Unique stations:", pairs["Provisional_Station_ID"].nunique())
print("\nSPLIT COUNTS:")
print(pairs["Split"].value_counts().to_string())
print("\nTARGET GAP (DAYS):")
print(pairs["Target_Gap_Days"].describe().round(2).to_string())
print("\nOUTPUT:", OUT)
print("SUMMARY:", SUMMARY)
print("\nFuture target gap is metadata only, not a model feature.")
