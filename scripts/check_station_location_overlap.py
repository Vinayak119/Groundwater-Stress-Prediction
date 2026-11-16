from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "reports" / "priority_station_records.csv"

df = pd.read_csv(src, low_memory=False)

df["Date"] = pd.to_datetime(
    df["Data Acquisition Time"], errors="coerce"
).dt.normalize()

df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce").round(5)
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce").round(5)

df["Location"] = (
    df["District"].fillna("UNKNOWN").astype(str) + " | " +
    df["Block"].fillna("UNKNOWN").astype(str) + " | " +
    df["Village"].fillna("UNKNOWN").astype(str) + " | " +
    df["Latitude"].astype(str) + " | " +
    df["Longitude"].astype(str)
)

rows = []

for station, group in df.groupby("Station"):
    locations = group["Location"].drop_duplicates().tolist()

    if len(locations) < 2:
        continue

    for i in range(len(locations)):
        for j in range(i + 1, len(locations)):
            a = group[group["Location"] == locations[i]]
            b = group[group["Location"] == locations[j]]

            dates_a = set(a["Date"].dropna())
            dates_b = set(b["Date"].dropna())
            overlap = dates_a.intersection(dates_b)

            rows.append({
                "Station": station,
                "Location_A": locations[i],
                "Location_B": locations[j],
                "Records_A": len(a),
                "Records_B": len(b),
                "Overlapping_Dates": len(overlap),
                "First_Overlap": min(overlap) if overlap else None,
                "Last_Overlap": max(overlap) if overlap else None
            })

result = pd.DataFrame(rows)

out = ROOT / "reports" / "station_location_overlap.csv"
result.to_csv(out, index=False)

print(result.to_string(index=False))
print("\nSAVED:", out)
