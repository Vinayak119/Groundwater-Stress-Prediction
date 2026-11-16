from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

src = ROOT / "data/processed/observations_location_keyed.csv"
out = ROOT / "reports"

df = pd.read_csv(src, low_memory=False)
df["Parsed_Date"] = pd.to_datetime(df["Parsed_Date"], errors="coerce")

keys = ["Provisional_Station_ID", "Parsed_Date"]

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

repeated["Conflict_Type"] = repeated["Distinct_Readings"].map(
    lambda n: "DIFFERENT_READINGS" if n > 1 else "SAME_READING"
)

repeated.to_csv(
    out / "all_repeated_station_date_groups.csv",
    index=False
)

repeated_rows = df.merge(
    repeated[keys],
    on=keys,
    how="inner"
)

repeated_rows.to_csv(
    out / "all_repeated_station_date_observations.csv",
    index=False
)

print("REPEATED STATION-DATE GROUPS:", len(repeated))
print("GROUPS WITH DIFFERENT READINGS:",
      int((repeated["Distinct_Readings"] > 1).sum()))
print("GROUPS WITH IDENTICAL READINGS:",
      int((repeated["Distinct_Readings"] == 1).sum()))
print("OBSERVATIONS IN REPEATED GROUPS:", len(repeated_rows))

print("\nGROUPS BY TYPE:")
print(repeated["Conflict_Type"].value_counts().to_string())

print("\nREPEATED GROUPS:")
print(repeated.to_string(index=False))

print("\nSAVED:")
print(out / "all_repeated_station_date_groups.csv")
print(out / "all_repeated_station_date_observations.csv")
