from pathlib import Path
import pandas as pd
import json

root = Path(__file__).resolve().parents[1]
source = root / "data/processed/forecasting_pairs_next_observation_150d.csv"
output = root / "data/processed/forecasting_pairs_temporally_purged.csv"

d = pd.read_csv(source)
d["Parsed_Date"] = pd.to_datetime(d["Parsed_Date"])
d["Target_Date"] = pd.to_datetime(d["Target_Date"])

train = d[d["Target_Date"] <= "2018-12-31"].copy()

validation = d[
    (d["Parsed_Date"] >= "2019-01-01") &
    (d["Target_Date"] <= "2020-12-31")
].copy()

test = d[d["Parsed_Date"] >= "2021-01-01"].copy()

train["Split"] = "train"
validation["Split"] = "validation"
test["Split"] = "test"

result = pd.concat([train, validation, test]).sort_values(
    ["Parsed_Date", "Provisional_Station_ID"]
).reset_index(drop=True)

result.to_csv(output, index=False)

summary = {
    "original_pairs": int(len(d)),
    "retained_pairs": int(len(result)),
    "excluded_boundary_pairs": int(len(d) - len(result)),
    "split_counts": {
        str(k): int(v)
        for k, v in result["Split"].value_counts().items()
    },
    "split_ranges": {
        str(name): {
            "input_start": str(g["Parsed_Date"].min()),
            "input_end": str(g["Parsed_Date"].max()),
            "target_start": str(g["Target_Date"].min()),
            "target_end": str(g["Target_Date"].max())
        }
        for name, g in result.groupby("Split")
    }
}

(root / "reports/temporal_split_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

print("CORRECTED TEMPORAL DATASET")
print("Original pairs:", len(d))
print("Retained pairs:", len(result))
print("Boundary pairs excluded:", len(d) - len(result))
print("\nSPLIT COUNTS:")
print(result["Split"].value_counts().to_string())
print("\nDATE RANGES:")
for name, group in result.groupby("Split"):
    print(
        f"{name}: input {group['Parsed_Date'].min()} to "
        f"{group['Parsed_Date'].max()} | target "
        f"{group['Target_Date'].min()} to {group['Target_Date'].max()}"
    )

print("\nSaved:", output)
