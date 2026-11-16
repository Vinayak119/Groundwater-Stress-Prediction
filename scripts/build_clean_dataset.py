from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

RAW_FILES = [
    ROOT / "data/raw/gwl_manual_quarterly_cgwb_tn_1991_2020.csv",
    ROOT / "data/raw/gwl_manual_quarterly_cgwb_tn_2021_2025.csv",
]

OUT = ROOT / "data/processed"
REPORTS = ROOT / "reports"

OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

DATE_COL = "Data Acquisition Time"
GWL_COL = "Groundwater Level Quarterly Manual (meter)"

# These station names have multiple reported geographic locations.
# Keys are based on the supplied district, block, village and coordinates.
SPECIAL_LOCATION_MAP = {
    ("Gudalur2", "Theni", "KAMBAM", "Gudalur", 9.70000, 77.30000):
        "Gudalur2__THENI_KAMBAM",
    ("Gudalur2", "Nilgiris", "GUDALUR", "Gudalur", 11.50917, 76.48278):
        "Gudalur2__NILGIRIS_GUDALUR",

    ("Kilpaur", "Tirunelveli", "Kilapavur", "Kilpaur", 8.91667, 77.38333):
        "Kilpaur__KILAPAVUR",
    ("Kilpaur", "Tirunelveli", "KADAYAM", "Kadayam", 8.83333, 77.40833):
        "Kilpaur__KADAYAM",

    ("Mettupalayam2", "Coimbatore", "KARAMADAI", "Mettupalayam", 11.30000, 76.94722):
        "Mettupalayam2__COIMBATORE",
    ("Mettupalayam2", "Cuddalore", "KURINCHIPADI", "Mettupalayam", 11.57500, 79.71528):
        "Mettupalayam2__CUDDALORE",

    ("Papparapatti", "Dharmapuri", "PENNAGARAM", "Papparapatti", 12.22000, 78.05944):
        "Papparapatti__DHARMAPURI",
    ("Papparapatti", "Salem", "SALEM", "Papparapatti", 11.52556, 78.05000):
        "Papparapatti__SALEM",

    ("Thanjavur", "Thanjavur", "THANJAVUR", "Thanjavur", 10.77500, 79.10833):
        "Thanjavur__THANJAVUR",
    ("Thanjavur", "Thanjavur", "BUDALUR", "Pudupatti", 10.85417, 78.95139):
        "Thanjavur__BUDALUR_PUDUPATTI",

    ("Vallam1", "Viluppuram", "VALLAM", "Vallam", 12.25278, 79.51250):
        "Vallam1__VALLAM",
    ("Vallam1", "Viluppuram", "MELMALAYANUR", "Devanur", 12.35278, 79.37500):
        "Vallam1__DEVANUR",
}

def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()

def make_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

frames = []
source_audit = []

for path in RAW_FILES:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw input: {path}")

    df = pd.read_csv(path, low_memory=False)

    required = [
        "Station", "District", "Block", "Village",
        "Latitude", "Longitude", DATE_COL, GWL_COL
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    df["Source_File"] = path.name
    df["Source_Row_Number"] = np.arange(2, len(df) + 2)

    source_audit.append({
        "file": path.name,
        "rows": int(len(df)),
        "sha256": make_sha256(path),
    })

    frames.append(df)

data = pd.concat(frames, ignore_index=True)
initial_rows = len(data)

# Parse values without silently changing the source columns.
data["Parsed_Date"] = pd.to_datetime(
    data[DATE_COL],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

data["GWL_m_bgl"] = pd.to_numeric(data[GWL_COL], errors="coerce")
data["Latitude_num"] = pd.to_numeric(data["Latitude"], errors="coerce")
data["Longitude_num"] = pd.to_numeric(data["Longitude"], errors="coerce")

for col in ["Station", "District", "Block", "Village"]:
    data[col + "_clean"] = data[col].map(clean_text)

data["Latitude_5dp"] = data["Latitude_num"].round(5)
data["Longitude_5dp"] = data["Longitude_num"].round(5)

# Assign provisional geographic keys.
# Non-special stations use their complete location metadata.
def assign_station_key(row):
    station = row["Station_clean"]
    district = row["District_clean"]
    block = row["Block_clean"]
    village = row["Village_clean"]
    lat = row["Latitude_5dp"]
    lon = row["Longitude_5dp"]

    if station == "Maruthur2":
        return "QUARANTINE__MARUTHUR2"

    special_key = (
        station, district, block, village, lat, lon
    )
    if special_key in SPECIAL_LOCATION_MAP:
        return SPECIAL_LOCATION_MAP[special_key]

    if not station:
        return "UNRESOLVED__MISSING_STATION"

    if pd.isna(lat) or pd.isna(lon):
        return (
            f"{station}__UNRESOLVED__"
            f"{district}_{block}_{village}"
        )

    return (
        f"{station}__{district}_{block}_{village}"
        f"__{lat:.5f}_{lon:.5f}"
    )

data["Provisional_Station_ID"] = data.apply(assign_station_key, axis=1)

# Build explicit exclusion reasons. No automatic deletion of negative,
# zero, or high readings.
data["Exclusion_Reason"] = ""

data.loc[
    data["Parsed_Date"].isna(),
    "Exclusion_Reason"
] = "INVALID_OR_UNPARSEABLE_DATE"

data.loc[
    data["GWL_m_bgl"].isna(),
    "Exclusion_Reason"
] = "MISSING_OR_INVALID_GROUNDWATER_READING"

data.loc[
    data["Station_clean"] == "Maruthur2",
    "Exclusion_Reason"
] = "QUARANTINED_AMBIGUOUS_WELL_IDENTITY"

# Specific anomaly identified in the supplied research investigation.
tiruvalangadu_anomaly = (
    data["Station_clean"].str.casefold().eq("tiruvalangadu")
    & data["Parsed_Date"].dt.strftime("%Y-%m-%d").eq("2019-01-30")
    & data["GWL_m_bgl"].eq(-97.0)
)

data.loc[
    tiruvalangadu_anomaly,
    "Exclusion_Reason"
] = "QUARANTINED_EXTREME_READING_PENDING_SOURCE_VERIFICATION"

# Quarantined records are preserved in a separate output.
quarantine = data[data["Exclusion_Reason"] != ""].copy()
eligible = data[data["Exclusion_Reason"] == ""].copy()

# Detect conflicting same-location same-date readings.
# Do not remove or average them at this stage.
conflict_keys = ["Provisional_Station_ID", "Parsed_Date"]

conflict_groups = (
    eligible.groupby(conflict_keys, dropna=False)
    .agg(
        Observation_Count=("GWL_m_bgl", "size"),
        Distinct_Readings=("GWL_m_bgl", "nunique"),
        Readings=("GWL_m_bgl", lambda x: " | ".join(map(str, x.tolist())))
    )
    .reset_index()
)

conflicts = conflict_groups[
    (conflict_groups["Observation_Count"] > 1)
    & (conflict_groups["Distinct_Readings"] > 1)
].copy()

# Keep all eligible records, including conflicts, for review.
eligible.to_csv(OUT / "observations_location_keyed.csv", index=False)
quarantine.to_csv(OUT / "quarantined_observations.csv", index=False)
conflicts.to_csv(OUT / "location_date_conflicts.csv", index=False)

# Record all source observations in a cleaning register.
register = quarantine[[
    "Source_File", "Source_Row_Number", "Station",
    "District", "Block", "Village", DATE_COL,
    GWL_COL, "Provisional_Station_ID", "Exclusion_Reason"
]].copy()

register.to_csv(OUT / "cleaning_exclusion_register.csv", index=False)

summary = {
    "source_files": source_audit,
    "initial_rows": int(initial_rows),
    "eligible_rows": int(len(eligible)),
    "quarantined_rows": int(len(quarantine)),
    "invalid_dates": int(data["Parsed_Date"].isna().sum()),
    "missing_or_invalid_gwl": int(data["GWL_m_bgl"].isna().sum()),
    "maruthur2_quarantined": int(
        (data["Station_clean"] == "Maruthur2").sum()
    ),
    "tiruvalangadu_minus_97_quarantined": int(tiruvalangadu_anomaly.sum()),
    "provisional_station_count": int(
        eligible["Provisional_Station_ID"].nunique()
    ),
    "location_date_conflict_groups": int(len(conflicts)),
    "negative_readings_retained": int((eligible["GWL_m_bgl"] < 0).sum()),
    "zero_readings_retained": int((eligible["GWL_m_bgl"] == 0).sum()),
    "positive_readings_retained": int((eligible["GWL_m_bgl"] > 0).sum()),
    "date_min": str(eligible["Parsed_Date"].min()),
    "date_max": str(eligible["Parsed_Date"].max()),
}

with open(REPORTS / "cleaning_pipeline_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nCLEANING PIPELINE SUMMARY")
print(json.dumps(summary, indent=2))
print("\nOUTPUT FILES")
print(OUT / "observations_location_keyed.csv")
print(OUT / "quarantined_observations.csv")
print(OUT / "location_date_conflicts.csv")
print(OUT / "cleaning_exclusion_register.csv")
print(REPORTS / "cleaning_pipeline_summary.json")
