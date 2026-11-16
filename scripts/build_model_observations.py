from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

src = ROOT / "data/processed/observations_location_keyed.csv"
out = ROOT / "data/processed"
reports = ROOT / "reports"

df = pd.read_csv(src, low_memory=False)

df["Parsed_Date"] = pd.to_datetime(df["Parsed_Date"], errors="coerce")
df["_Calendar_Date"] = df["Parsed_Date"].dt.normalize()
df["_Source_Row"] = range(len(df))

keys = ["Provisional_Station_ID", "_Calendar_Date"]

grouped = (
    df.groupby(keys, dropna=False)
      .agg(
          Observation_Count=("GWL_m_bgl", "size"),
          Distinct_Readings=("GWL_m_bgl", "nunique"),
          Readings=("GWL_m_bgl",
                    lambda x: " | ".join(map(str, x.tolist())))
      )
      .reset_index()
)

repeated = grouped[grouped["Observation_Count"] > 1].copy()

repeated_ids = set(
    df.merge(repeated[keys], on=keys, how="inner")["_Source_Row"]
)

quarantine = df[df["_Source_Row"].isin(repeated_ids)].copy()
model_ready = df[~df["_Source_Row"].isin(repeated_ids)].copy()

quarantine["Quarantine_Reason"] = (
    "Multiple observations for the same provisional station and calendar date"
)

quarantine["Audit_Calendar_Date"] = quarantine["_Calendar_Date"]

quarantine = quarantine.drop(columns=["_Calendar_Date", "_Source_Row"])
model_ready = model_ready.drop(columns=["_Calendar_Date", "_Source_Row"])

quarantine_path = out / "observations_quarantined.csv"
model_path = out / "observations_model_ready.csv"
report_path = reports / "calendar_date_conflict_groups.csv"

quarantine.to_csv(quarantine_path, index=False)
model_ready.to_csv(model_path, index=False)
repeated.to_csv(report_path, index=False)

# Verify that no repeated station/calendar-date groups remain.
check = model_ready.copy()
check["_Calendar_Date"] = pd.to_datetime(
    check["Parsed_Date"], errors="coerce"
).dt.normalize()

remaining = (
    check.groupby(
        ["Provisional_Station_ID", "_Calendar_Date"],
        dropna=False
    ).size()
)

remaining = remaining[remaining > 1]

print("\nDATASET BUILD COMPLETE")
print("Original eligible observations:", len(df))
print("Quarantined observations:", len(quarantine))
print("Model-ready observations:", len(model_ready))
print("Repeated station-date groups quarantined:", len(repeated))
print("Repeated groups remaining in model-ready data:", len(remaining))
print("Rows with missing station IDs:", model_ready["Provisional_Station_ID"].isna().sum())
print("Rows with missing dates:", model_ready["Parsed_Date"].isna().sum())

print("\nOUTPUT FILES:")
print(model_path)
print(quarantine_path)
print(report_path)

if len(remaining) == 0:
    print("\nSUCCESS: No repeated station/calendar-date groups remain.")
else:
    print("\nWARNING: Repeated groups remain. Do not start model training.")
