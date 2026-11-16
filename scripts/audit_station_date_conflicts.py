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

    frames.append(df)

data = pd.concat(frames, ignore_index=True)

keys = ["Station", "Parsed_Date"]

summary = (
    data.groupby(keys, dropna=False)
    .agg(
        observation_count=(TARGET, "size"),
        distinct_readings=(TARGET, "nunique"),
        readings=(TARGET, lambda x: " | ".join(
            f"{v:.2f}" for v in sorted(x.dropna())
        )),
        districts=("District", "nunique"),
        latitude_count=("Latitude", "nunique"),
        longitude_count=("Longitude", "nunique"),
    )
    .reset_index()
)

conflicts = summary[
    (summary["observation_count"] > 1)
    & (summary["distinct_readings"] > 1)
].copy()

conflicts = conflicts.sort_values(
    ["observation_count", "Station"], ascending=[False, True]
)

output = REPORTS / "station_date_conflicts.csv"
conflicts.to_csv(output, index=False)

print("TOTAL RAW ROWS:", len(data))
print("DUPLICATE STATION-DATE GROUPS:", int(
    (summary["observation_count"] > 1).sum()
))
print("CONFLICTING STATION-DATE GROUPS:", len(conflicts))
print("\nCONFLICTS BY DATASET PERIOD:")

for filename in files:
    dates = conflicts[
        conflicts["Parsed_Date"].isin(
            data.loc[data["Source_File"] == filename, "Parsed_Date"]
        )
    ]
    print(filename, "groups:", len(dates))

print("\nFIRST 30 CONFLICTS")
print(conflicts.head(30).to_string(index=False))

print("\nSaved:", output)
