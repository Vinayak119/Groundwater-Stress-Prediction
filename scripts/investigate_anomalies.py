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

reports = []

for filename in FILES:
    print(f"\nInvestigating {filename}")

    df = pd.read_csv(RAW / filename, low_memory=False)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    df["Dataset"] = filename
    df["Parsed_Date"] = pd.to_datetime(
        df[DATE],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    print("Rows:", len(df))
    print("Negative readings:", (df[TARGET] < 0).sum())
    print("Missing readings:", df[TARGET].isna().sum())

    review = pd.concat([
        df.nsmallest(20, TARGET),
        df.nlargest(20, TARGET),
        df[df[TARGET] < 0],
        df[df[TARGET].isna()]
    ]).drop_duplicates()

    preferred = [
        "Dataset", "Station", "District", "Block", "Village",
        "Latitude", "Longitude", DATE, "Parsed_Date",
        TARGET, "RL_MSL"
    ]

    columns = [c for c in preferred if c in review.columns]
    reports.append(review[columns])

    print("\nLowest readings:")
    print(
        df.nsmallest(10, TARGET)[columns]
        .to_string(index=False)
    )

    print("\nHighest readings:")
    print(
        df.nlargest(10, TARGET)[columns]
        .to_string(index=False)
    )

combined = pd.concat(reports, ignore_index=True)

output = REPORTS / "anomaly_investigation.csv"
combined.to_csv(output, index=False)

print("\nInvestigation report saved:", output)
print("Total records flagged for review:", len(combined))
