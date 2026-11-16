from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "data/processed/forecasting_pairs_temporally_purged.csv")
model = joblib.load(root / "models/trained/lightgbm.joblib")

features = [
    "GWL_m_bgl",
    "Previous_GWL_m_bgl",
    "Two_Observations_Back_GWL_m_bgl",
    "Historical_Mean_GWL_m_bgl",
    "Historical_Std_GWL_m_bgl",
    "History_Count",
    "Days_Since_Previous_Observation",
    "Current_Month",
    "Month_Sin",
    "Month_Cos",
    "Latitude_num",
    "Longitude_num"
]

test = df[df["Split"] == "test"].copy()
test["Target_GWL_m_bgl"] = pd.to_numeric(test["Target_GWL_m_bgl"])
test["LightGBM_Prediction"] = model.predict(test[features])
test["Persistence_Prediction"] = test["GWL_m_bgl"]
test["LightGBM_Absolute_Error"] = (
    test["Target_GWL_m_bgl"] - test["LightGBM_Prediction"]
).abs()
test["Persistence_Absolute_Error"] = (
    test["Target_GWL_m_bgl"] - test["Persistence_Prediction"]
).abs()

test["Gap_Group"] = pd.cut(
    test["Target_Gap_Days"],
    bins=[0, 60, 90, 120, 150],
    labels=["0-60 days", "61-90 days", "91-120 days", "121-150 days"],
    include_lowest=True
)

rows = []
for group, part in test.groupby("Gap_Group", observed=True):
    for name, pred_col in [
        ("Persistence", "Persistence_Prediction"),
        ("LightGBM", "LightGBM_Prediction")
    ]:
        y = part["Target_GWL_m_bgl"]
        p = part[pred_col]
        rows.append({
            "Gap_Group": str(group),
            "Model": name,
            "Rows": len(part),
            "MAE": mean_absolute_error(y, p),
            "RMSE": np.sqrt(mean_squared_error(y, p)),
            "R2": r2_score(y, p)
        })

result = pd.DataFrame(rows)
out = root / "results/metrics/test_performance_by_gap.csv"
result.to_csv(out, index=False)

print("TEST PERFORMANCE BY OBSERVATION GAP")
print(result.round(4).to_string(index=False))

print("\nMEAN ABSOLUTE ERROR IMPROVEMENT")
comparison = test.groupby("Gap_Group", observed=True).agg(
    Persistence_MAE=("Persistence_Absolute_Error", "mean"),
    LightGBM_MAE=("LightGBM_Absolute_Error", "mean"),
    Rows=("Target_GWL_m_bgl", "size")
)
comparison["LightGBM_Improvement_Percent"] = (
    (comparison["Persistence_MAE"] - comparison["LightGBM_MAE"])
    / comparison["Persistence_MAE"] * 100
)
print(comparison.round(3).to_string())

print("\nSAVED:", out)
