from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ROOT / "data" / "raw" / "gwl_manual_quarterly_cgwb_tn_1991_2020.csv",
    ROOT / "data" / "raw" / "gwl_manual_quarterly_cgwb_tn_2021_2025.csv",
]

stations = [
    "Gudalur2", "Vallam1", "Thanjavur", "Kilpaur",
    "Mettupalayam2", "Papparapatti", "Maruthur2"
]

frames = []
for path in FILES:
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    if "Station" not in df.columns:
        raise ValueError(f"'Station' column missing in {path.name}")

    df["Source_File"] = path.name
    selected = df[df["Station"].astype(str).str.strip().isin(stations)].copy()
    frames.append(selected)

result = pd.concat(frames, ignore_index=True)

out = ROOT / "reports" / "priority_station_records.csv"
result.to_csv(out, index=False)

print("SELECTED RECORDS:", len(result))
print("STATION COUNTS:")
print(result["Station"].value_counts().to_string())
print("\nAVAILABLE COLUMNS:")
print(", ".join(result.columns))
print("\nSAVED:", out)
