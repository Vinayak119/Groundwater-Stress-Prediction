from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    root / "data/processed/forecasting_pairs_temporally_purged.csv"
)
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

validation = df[df["Split"] == "validation"].copy()
sample = validation.sample(
    n=min(1000, len(validation)), random_state=42
)

imputer = model.named_steps["imputer"]
estimator = model.named_steps["model"]

X = imputer.transform(sample[features])
names = imputer.get_feature_names_out(features)
X = pd.DataFrame(X, columns=names, index=sample.index)

print("Calculating SHAP values for", len(sample), "validation records...")

explainer = shap.TreeExplainer(estimator)
values = explainer.shap_values(X)

if isinstance(values, list):
    values = values[0]

values = np.asarray(values)

if values.ndim == 3:
    values = values[:, :, 0]

importance = pd.DataFrame({
    "Feature": names,
    "Mean_Absolute_SHAP": np.abs(values).mean(axis=0)
}).sort_values("Mean_Absolute_SHAP", ascending=False)

importance["Importance_Percent"] = (
    importance["Mean_Absolute_SHAP"] /
    importance["Mean_Absolute_SHAP"].sum() * 100
)

out_dir = root / "results/shap"
out_dir.mkdir(parents=True, exist_ok=True)

importance.to_csv(out_dir / "shap_feature_importance.csv", index=False)

plt.figure()
shap.summary_plot(values, X, show=False, max_display=15)
plt.tight_layout()
plt.savefig(out_dir / "shap_summary.png", dpi=180, bbox_inches="tight")
plt.close()

plt.figure()
shap.summary_plot(values, X, plot_type="bar", show=False, max_display=15)
plt.tight_layout()
plt.savefig(out_dir / "shap_bar.png", dpi=180, bbox_inches="tight")
plt.close()

print("\nSHAP FEATURE IMPORTANCE")
print(importance.round(3).to_string(index=False))
print("\nSaved SHAP outputs in:", out_dir)
