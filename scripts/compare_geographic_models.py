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

districts = [
    "Kanyakumari", "Karur", "Perambalur",
    "Ramanathapuram", "Sivaganga", "Tiruvallur", "Viluppuram"
]

test = df[(df.Split == "test") & df.District.isin(districts)].copy()
test["LightGBM_Prediction"] = model.predict(test[features])
test["Persistence_Prediction"] = test["GWL_m_bgl"]

rows = []
for district, g in test.groupby("District"):
    y = g["Target_GWL_m_bgl"]
    for name, col in [
        ("Persistence", "Persistence_Prediction"),
        ("LightGBM", "LightGBM_Prediction")
    ]:
        err = (y - g[col]).abs()
        rows.append({
            "District": district,
            "Model": name,
            "Predictions": len(g),
            "MAE": err.mean(),
            "RMSE": np.sqrt(((y - g[col]) ** 2).mean()),
            "Maximum_Absolute_Error": err.max()
        })

result = pd.DataFrame(rows)
result.to_csv(root / "results/metrics/geographic_model_comparison_by_district.csv", index=False)

print("MODEL COMPARISON BY DISTRICT")
print(result.round(3).to_string(index=False))

print("\nLARGEST LIGHTGBM ERRORS")
test["Absolute_Error"] = (
    test["Target_GWL_m_bgl"] - test["LightGBM_Prediction"]
).abs()

cols = [
    "District", "Station", "Parsed_Date", "Target_Date",
    "GWL_m_bgl", "Target_GWL_m_bgl",
    "LightGBM_Prediction", "Absolute_Error"
]
print(test.nlargest(10, "Absolute_Error")[cols].round(3).to_string(index=False))

print("\nSaved: results/metrics/geographic_model_comparison_by_district.csv")
