from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

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

target = "Target_GWL_m_bgl"
train = df[df.Split == "train"]
val = df[df.Split == "validation"]

X_train, y_train = train[features], train[target]
X_val, y_val = val[features], val[target]

configs = [
    {"num_leaves": 15, "max_depth": 5, "min_child_samples": 20, "reg_lambda": 2},
    {"num_leaves": 31, "max_depth": 7, "min_child_samples": 20, "reg_lambda": 2},
    {"num_leaves": 15, "max_depth": -1, "min_child_samples": 40, "reg_lambda": 5},
    {"num_leaves": 31, "max_depth": 10, "min_child_samples": 40, "reg_lambda": 5}
]

results = []
best_model = None
best_mae = float("inf")

# Persistence reference on validation
baseline = X_val["GWL_m_bgl"]
baseline_mae = mean_absolute_error(y_val, baseline)
baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline))

results.append({
    "Model": "Persistence",
    "Validation_MAE": baseline_mae,
    "Validation_RMSE": baseline_rmse,
    "Validation_R2": r2_score(y_val, baseline)
})

for i, config in enumerate(configs, 1):
    print(f"Training configuration {i}/4: {config}", flush=True)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            n_jobs=4,
            random_state=42,
            verbosity=-1,
            **config
        ))
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    r2 = r2_score(y_val, predictions)

    results.append({
        "Model": f"LightGBM_Config_{i}",
        "Validation_MAE": mae,
        "Validation_RMSE": rmse,
        "Validation_R2": r2
    })

    if mae < best_mae:
        best_mae = mae
        best_model = pipeline

    print(f"Validation MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

out = pd.DataFrame(results).sort_values("Validation_MAE")
out_path = root / "results/metrics/lightgbm_validation_tuning.csv"
out.to_csv(out_path, index=False)

if best_model is not None:
    model_path = root / "models/trained/lightgbm_tuned.joblib"
    joblib.dump(best_model, model_path)
    print("\nBest LightGBM saved:", model_path)

print("\nVALIDATION COMPARISON")
print(out.round(4).to_string(index=False))
print("\nPersistence validation MAE:", round(baseline_mae, 4))
print("Best LightGBM validation MAE:", round(best_mae, 4))
print("Results saved:", out_path)
