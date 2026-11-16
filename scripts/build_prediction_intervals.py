from pathlib import Path
import numpy as np
import pandas as pd
import joblib

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "data/processed/forecasting_pairs_temporally_purged.csv")

features = [
    "GWL_m_bgl", "Previous_GWL_m_bgl",
    "Two_Observations_Back_GWL_m_bgl",
    "Historical_Mean_GWL_m_bgl", "Historical_Std_GWL_m_bgl",
    "History_Count", "Days_Since_Previous_Observation",
    "Current_Month", "Month_Sin", "Month_Cos",
    "Latitude_num", "Longitude_num"
]

model = joblib.load(root / "models/trained/lightgbm_tuned.joblib")
validation = df[df["Split"] == "validation"].copy()
test = df[df["Split"] == "test"].copy()

# Estimate the error range from validation residuals
val_pred = model.predict(validation[features])
val_errors = np.abs(validation["Target_GWL_m_bgl"] - val_pred)

alpha = 0.10
n = len(val_errors)
q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
radius = float(np.quantile(val_errors, q_level))

# Apply the same interval width to future test predictions
test["Prediction"] = model.predict(test[features])
test["Lower_90"] = test["Prediction"] - radius
test["Upper_90"] = test["Prediction"] + radius
test["Actual"] = test["Target_GWL_m_bgl"]
test["Covered"] = (
    (test["Actual"] >= test["Lower_90"]) &
    (test["Actual"] <= test["Upper_90"])
)

print("PRELIMINARY 90% PREDICTION INTERVAL")
print("Validation records:", n)
print("Interval radius (+/- metres):", round(radius, 3))
print("Test records:", len(test))
print("Test coverage:", round(test["Covered"].mean() * 100, 2), "%")
print("Average interval width:", round(2 * radius, 3), "metres")

test["Gap_Group"] = pd.cut(
    test["Target_Gap_Days"],
    bins=[0, 60, 90, 120, 150],
    labels=["0-60 days", "61-90 days", "91-120 days", "121-150 days"],
    include_lowest=True
)

print("\nCOVERAGE BY OBSERVATION GAP")
print(
    test.groupby("Gap_Group", observed=True)
    .agg(
        Records=("Covered", "size"),
        Coverage=("Covered", "mean")
    )
    .assign(Coverage_Percent=lambda x: x["Coverage"] * 100)
    .drop(columns="Coverage")
    .round(2)
    .to_string()
)

out_dir = root / "results/metrics"
test.to_csv(out_dir / "preliminary_prediction_intervals.csv", index=False)

print("\nSaved:", out_dir / "preliminary_prediction_intervals.csv")
print("NOTE: Exploratory intervals; validation data was also used for model selection.")
