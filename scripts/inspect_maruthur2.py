from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

TARGET = "Groundwater Level Quarterly Manual (meter)"
DATE = "Data Acquisition Time"

files = [
    "gwl_manual_quarterly_cgwb_tn_1991_2020.csv",
    "gwl_manual_quarterly_cgwb_tn_2021_2025.csv",
]

frames = []

for filename in files:
    df = pd.read_csv(RAW / filename, low_memory=False)
    df["Source_File"] = filename
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df["Parsed_Date"] = pd.to_datetime(
        df[DATE], format="%d-%m-%Y %H:%M", errors="coerce"
    )

    mask = df["Station"].astype(str).str.strip().str.lower().eq("maruthur2")
    frames.append(df.loc[mask].copy())

result = pd.concat(frames, ignore_index=True)
result = result.sort_values("Parsed_Date")

columns = [
    "Source_File", "Station", "District", "Block", "Village",
    "Latitude", "Longitude", DATE, "Parsed_Date", TARGET, "RL_MSL"
]
columns = [c for c in columns if c in result.columns]

print("MARUTHUR2 RECORDS")
print("Total records:", len(result))
print(result[columns].to_string(index=False))

output = REPORTS / "maruthur2_history.csv"
result[columns].to_csv(output, index=False)

print("\nSaved:", output)
