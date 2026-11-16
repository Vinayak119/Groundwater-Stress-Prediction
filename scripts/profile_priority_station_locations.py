from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "reports" / "priority_station_records.csv"
out = ROOT / "reports" / "priority_station_location_profile.csv"

df = pd.read_csv(src, low_memory=False)

date_col = "Data Acquisition Time"
gwl_col = "Groundwater Level Quarterly Manual (meter)"

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df[gwl_col] = pd.to_numeric(df[gwl_col], errors="coerce")
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

df["Latitude_5dp"] = df["Latitude"].round(5)
df["Longitude_5dp"] = df["Longitude"].round(5)

keys = [
    "Station", "District", "Block", "Village",
    "Latitude_5dp", "Longitude_5dp"
]

profile = (
    df.groupby(keys, dropna=False)
      .agg(
          Records=(gwl_col, "count"),
          First_Date=(date_col, "min"),
          Last_Date=(date_col, "max"),
          Distinct_Dates=(date_col, "nunique"),
          Min_GWL=(gwl_col, "min"),
          Median_GWL=(gwl_col, "median"),
          Max_GWL=(gwl_col, "max"),
          Distinct_GWL=(gwl_col, "nunique"),
          Distinct_RL_MSL=("RL_MSL", "nunique")
      )
      .reset_index()
      .sort_values(["Station", "Records"], ascending=[True, False])
)

profile.to_csv(out, index=False)

print("LOCATION GROUPS:", len(profile))
print("STATIONS:", profile["Station"].nunique())
print("\nLOCATION PROFILE:")
print(profile.to_string(index=False))
print("\nSAVED:", out)
