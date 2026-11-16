from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

TARGET = "Groundwater Level Quarterly Manual (meter)"
DATE = "Data Acquisition Time"

FILES = [
    "gwl_manual_quarterly_cgwb_tn_1991_2020.csv",
    "gwl_manual_quarterly_cgwb_tn_2021_2025.csv",
]

WATCH_STATIONS = {
    "Tiruvalangadu",
    "Kopuvanur1",
    "Panamarathupatti",
    "Pudukurayapetai",
    "Irruppu",
    "T.Krishnapuram",
    "Choladasanpatti pz",
    "Vinnampalli",
    "Marungur-II OW",
    "C.Keeranur EW",
    "Marungur -I EW",
    "ENUSONAI_2",
    "Kadavur colony (D. Edayapatti East) OW",
    "Mahimapuram OW-2",
}

frames = []

for filename in FILES:
    path = RAW / filename
    df = pd.read_csv(path, low_memory=False)

    df["Source_File"] = filename
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df["Parsed_Date"] = pd.to_datetime(
        df[DATE], format="%d-%m-%Y %H:%M", errors="coerce"
    )

    frames.append(df)

data = pd.concat(frames, ignore_index=True)

# 1. Full histories of named stations and every record >= 50 m or <= -10 m.
station_mask = data["Station"].astype(str).isin(WATCH_STATIONS)
extreme_mask = (data[TARGET] >= 50) | (data[TARGET] <= -10)

history = data[station_mask | extreme_mask].copy()

preferred = [
    "Source_File", "Station", "District", "Block", "Village",
    "Latitude", "Longitude", DATE, "Parsed_Date",
    TARGET, "RL_MSL"
]
history_columns = [c for c in preferred if c in history.columns]

history = history[history_columns].sort_values(
    ["Station", "Parsed_Date"], na_position="last"
)

history_path = REPORTS / "extreme_reading_history.csv"
history.to_csv(history_path, index=False)

# 2. All missing groundwater readings.
missing = data[data[TARGET].isna()].copy()
missing_columns = [c for c in preferred if c in missing.columns]
missing_path = REPORTS / "missing_reading_records.csv"
missing[missing_columns].to_csv(missing_path, index=False)

# 3. Summary of stations with readings <= -10 m or >= 50 m.
flagged = data[extreme_mask].copy()

summary = (
    flagged.groupby(["Source_File", "Station", "District"], dropna=False)
    .agg(
        flagged_readings=(TARGET, "size"),
        minimum_reading=(TARGET, "min"),
        maximum_reading=(TARGET, "max"),
        first_flagged_date=("Parsed_Date", "min"),
        last_flagged_date=("Parsed_Date", "max"),
    )
    .reset_index()
    .sort_values(["Source_File", "maximum_reading"], ascending=[True, False])
)

summary_path = REPORTS / "station_anomaly_summary.csv"
summary.to_csv(summary_path, index=False)

print("\nSTATION HISTORY REPORT")
print("Rows:", len(history))
print("Saved:", history_path)

print("\nMISSING READING REPORT")
print("Rows:", len(missing))
print("Saved:", missing_path)

print("\nEXTREME STATION SUMMARY")
print(summary.to_string(index=False))
print("\nSaved:", summary_path)
