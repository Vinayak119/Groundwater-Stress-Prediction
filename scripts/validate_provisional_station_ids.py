from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/observations_location_keyed.csv"
CONFLICTS = ROOT / "data/processed/location_date_conflicts.csv"
OUT = ROOT / "reports"

df = pd.read_csv(DATA, low_memory=False)
conflicts = pd.read_csv(CONFLICTS, low_memory=False)

# Profile every provisional station ID.
profile = (
    df.groupby("Provisional_Station_ID")
      .agg(
          Records=("GWL_m_bgl", "size"),
          Distinct_Dates=("Parsed_Date", "nunique"),
          First_Date=("Parsed_Date", "min"),
          Last_Date=("Parsed_Date", "max"),
          Distinct_Districts=("District", "nunique"),
          Distinct_Latitudes=("Latitude_5dp", "nunique"),
          Distinct_Longitudes=("Longitude_5dp", "nunique"),
          Min_GWL=("GWL_m_bgl", "min"),
          Median_GWL=("GWL_m_bgl", "median"),
          Max_GWL=("GWL_m_bgl", "max")
      )
      .reset_index()
)

profile.to_csv(OUT / "provisional_station_id_profile.csv", index=False)

# Count IDs per original station name.
ids_per_name = (
    df.groupby("Station_clean")["Provisional_Station_ID"]
      .nunique()
      .sort_values(ascending=False)
)

# Find very small IDs for review.
small_ids = profile[profile["Records"] < 5].sort_values("Records")
small_ids.to_csv(OUT / "small_provisional_station_ids.csv", index=False)

# Exact duplicate observations in the eligible dataset.
comparison_cols = [
    "Station_clean", "Parsed_Date", "GWL_m_bgl",
    "District", "Block", "Village",
    "Latitude_5dp", "Longitude_5dp"
]
exact_duplicates = df[
    df.duplicated(subset=comparison_cols, keep=False)
].sort_values(["Station_clean", "Parsed_Date"])

exact_duplicates.to_csv(OUT / "eligible_exact_duplicate_observations.csv", index=False)

print("ELIGIBLE ROWS:", len(df))
print("UNIQUE PROVISIONAL IDS:", profile.shape[0])
print("IDS WITH FEWER THAN 5 RECORDS:", len(small_ids))
print("EXACT DUPLICATE OBSERVATION ROWS:", len(exact_duplicates))
print("REMAINING CONFLICT GROUPS:", len(conflicts))

print("\nRECORDS PER ID DISTRIBUTION:")
print(profile["Records"].describe().to_string())

print("\nORIGINAL STATION NAMES WITH MOST PROVISIONAL IDS:")
print(ids_per_name.head(20).to_string())

print("\nSMALLEST PROVISIONAL IDS:")
print(small_ids.head(20).to_string(index=False))

print("\nREMAINING CONFLICTS:")
print(conflicts.to_string(index=False))

print("\nSAVED REPORTS:")
print(OUT / "provisional_station_id_profile.csv")
print(OUT / "small_provisional_station_ids.csv")
print(OUT / "eligible_exact_duplicate_observations.csv")
