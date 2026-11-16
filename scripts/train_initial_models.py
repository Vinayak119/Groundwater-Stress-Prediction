from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/forecasting_pairs_temporally_purged.csv"
MODEL_DIR = ROOT / "models/trained"
RESULT_DIR = ROOT / "results/metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
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

TARGET = "Target_GWL_m_bgl"

df = pd.read_csv(DATA)

for col in FEATURES + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

train = df[df["Split"] == "train"].copy()
validation = df[df["Split"] == "validation"].copy()
test = df[df["Split"] == "test"].copy()

X_train, y_train = train[FEATURES], train[TARGET]
X_val, y_val = validation[FEATURES], validation[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

models = {
    "Decision Tree": DecisionTreeRegressor(
        max_depth=15, min_samples_leaf=5, random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=250, min_samples_leaf=2,
        max_features=0.9, n_jobs=-1, random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=2, objective="reg:squarederror",
        n_jobs=4, random_state=42, verbosity=0
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=300, max_depth=7, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=2, n_jobs=4, random_state=42, verbosity=-1
    )
}

def evaluate(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
        "R2": r2_score(y_true, predictions)
    }

results = []

# Persistence baseline: predict that the next reading equals the current one
for split_name, X, y in [
    ("validation", X_val, y_val),
    ("test", X_test, y_test)
]:
    score = evaluate(y, X["GWL_m_bgl"])
    results.append({
        "Model": "Persistence Baseline",
        "Split": split_name,
        **score
    })

# Fit all learned preprocessing on training data only
for name, estimator in models.items():
    print(f"\nTraining {name}...", flush=True)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", estimator)
    ])

    try:
        pipeline.fit(X_train, y_train)

        for split_name, X, y in [
            ("validation", X_val, y_val),
            ("test", X_test, y_test)
        ]:
            predictions = pipeline.predict(X)
            score = evaluate(y, predictions)

            results.append({
                "Model": name,
                "Split": split_name,
                **score
            })

        model_path = MODEL_DIR / (
            name.lower().replace(" ", "_") + ".joblib"
        )
        joblib.dump(pipeline, model_path)
        print("Saved:", model_path, flush=True)

    except Exception as exc:
        print(f"FAILED {name}: {exc}", flush=True)

result_df = pd.DataFrame(results)
result_df.to_csv(RESULT_DIR / "initial_model_benchmark.csv", index=False)

summary = {
    "training_rows": int(len(train)),
    "validation_rows": int(len(validation)),
    "test_rows": int(len(test)),
    "features": FEATURES,
    "target": TARGET,
    "imputation": "Median imputation fitted on training data only",
    "test_used_for_training": False,
    "results": results
}

(ROOT / "reports/initial_model_benchmark.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

print("\n========== INITIAL MODEL BENCHMARK ==========")
print(result_df.round(4).to_string(index=False))
print("\nSaved metrics:", RESULT_DIR / "initial_model_benchmark.csv")
print("Saved report:", ROOT / "reports/initial_model_benchmark.json")
print("\nBenchmark complete.")
