from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

files = [
    ROOT / "data/raw/gwl_manual_quarterly_cgwb_tn_1991_2020.csv",
    ROOT / "data/raw/gwl_manual_quarterly_cgwb_tn_2021_2025.csv",
]

stations = [
    "Dharapuram2",
    "Kurinjipadi",
    "Ramanathapuram2",
    "Sethiatope2",
]

frames = []

for path in files:
    df = pd.read_csv(path, low_memory=False)
    df["Source_File"] = path.name
    df["Parsed_Date"] = pd.to_datetime(
        df["Data Acquisition Time"],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    selected = df[
        df["Station"].astype(str).str.strip().isin(stations)
    ].copy()

    frames.append(selected)

result = pd.concat(frames, ignore_index=True)

columns = [
    "Station", "District", "Block", "Village",
    "Latitude", "Longitude", "RL_MSL",
    "Parsed_Date",
    "Groundwater Level Quarterly Manual (meter)",
    "Source_File", "SlNo"
]

result = result[columns].sort_values(
    ["Station", "Parsed_Date", "Source_File"]
)

out = ROOT / "reports" / "remaining_conflict_source_records.csv"
result.to_csv(out, index=False)

print("RECORDS EXPORTED:", len(result))
print(result.to_string(index=False))
print("\nSAVED:", out)
