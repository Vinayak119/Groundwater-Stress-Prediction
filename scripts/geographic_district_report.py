from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "data/processed/forecasting_pairs_temporally_purged.csv")
model = joblib.load(root / "models/trained/lightgbm_geographic_holdout.joblib")

features = [
    "GWL_m_bgl", "Previous_GWL_m_bgl",
    "Two_Observations_Back_GWL_m_bgl",
    "Historical_Mean_GWL_m_bgl", "Historical_Std_GWL_m_bgl",
    "History_Count", "Days_Since_Previous_Observation",
    "Current_Month", "Month_Sin", "Month_Cos",
    "Latitude_num", "Longitude_num"
]

heldout_districts = [
    "Kanyakumari", "Karur", "Perambalur",
    "Ramanathapuram", "Sivaganga", "Tiruvallur", "Viluppuram"
]

test = df[
    (df["Split"] == "test") &
    (df["District"].isin(heldout_districts))
].copy()

test["Prediction"] = model.predict(test[features])
test["Absolute_Error"] = (
    test["Target_GWL_m_bgl"] - test["Prediction"]
).abs()

rows = []
for district, group in test.groupby("District"):
    rows.append({
        "District": district,
        "Stations": group["Provisional_Station_ID"].nunique(),
        "Predictions": len(group),
        "MAE": mean_absolute_error(
            group["Target_GWL_m_bgl"], group["Prediction"]
        ),
        "RMSE": np.sqrt(mean_squared_error(
            group["Target_GWL_m_bgl"], group["Prediction"]
        ))
    })

result = pd.DataFrame(rows).sort_values("MAE")
out = root / "results/metrics/geographic_district_performance.csv"
result.to_csv(out, index=False)

print("HELD-OUT DISTRICT PERFORMANCE")
print(result.round(3).to_string(index=False))
print("\nSaved:", out)
