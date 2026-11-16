from pathlib import Path
import pandas as pd
import joblib

root = Path(__file__).resolve().parents[1]
model = joblib.load(root / "models/trained/lightgbm_tuned.joblib")

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

imputer = model.named_steps["imputer"]
estimator = model.named_steps["model"]

actual_names = imputer.get_feature_names_out(features)
actual_importance = estimator.feature_importances_

importance = pd.DataFrame({
    "Feature": actual_names,
    "Importance": actual_importance
})

total = importance["Importance"].sum()

importance["Importance_Percent"] = (
    importance["Importance"] / total * 100 if total else 0
)

importance = importance.sort_values(
    "Importance_Percent", ascending=False
)

out = root / "results/metrics/lightgbm_feature_importance.csv"
importance.to_csv(out, index=False)

print("FEATURES USED BY TRAINED MODEL:", len(actual_names))
print("\nLIGHTGBM FEATURE IMPORTANCE")
print(importance.round(2).to_string(index=False))
print("\nSaved:", out)
