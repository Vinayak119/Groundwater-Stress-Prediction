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

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df["Parsed_Date"] = pd.to_datetime(
        df[DATE], format="%d-%m-%Y %H:%M", errors="coerce"
    )
    df["Source_File"] = filename

    mask = df["Station"].astype(str).str.strip().str.lower().eq(
        "maruthur2"
    )
    frames.append(df.loc[mask].copy())

data = pd.concat(frames, ignore_index=True)

# Count observations for each station-date.
date_summary = (
    data.groupby(["Parsed_Date"], dropna=False)
    .agg(
        observation_count=(TARGET, "size"),
        distinct_readings=(TARGET, "nunique"),
        readings=(TARGET, lambda x: " | ".join(
            f"{v:.2f}" for v in sorted(x.dropna())
        )),
    )
    .reset_index()
    .sort_values("Parsed_Date")
)

duplicates = date_summary[
    date_summary["observation_count"] > 1
].copy()

# Exact duplicate rows across original columns.
original_columns = [
    c for c in data.columns
    if c not in ["Parsed_Date", "Source_File"]
]

exact_duplicates = data[
    data.duplicated(subset=original_columns, keep=False)
].copy()

print("TOTAL MARUTHUR2 RECORDS:", len(data))
print("UNIQUE DATES:", data["Parsed_Date"].nunique())
print("DATES WITH MULTIPLE OBSERVATIONS:", len(duplicates))
print("EXACT DUPLICATE ROWS:", len(exact_duplicates))

print("\nDATE-LEVEL OBSERVATION SUMMARY")
print(duplicates.to_string(index=False))

print("\nEXACT DUPLICATE RECORDS")
if len(exact_duplicates):
    print(exact_duplicates[
        ["Source_File", DATE, TARGET]
    ].to_string(index=False))
else:
    print("No exact duplicate records found.")

date_summary.to_csv(
    REPORTS / "maruthur2_date_summary.csv", index=False
)
exact_duplicates.to_csv(
    REPORTS / "maruthur2_exact_duplicates.csv", index=False
)

print("\nReports saved in:", REPORTS)
