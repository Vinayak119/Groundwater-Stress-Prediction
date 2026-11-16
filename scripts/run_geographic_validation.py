from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit
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

df = df.dropna(subset=["District"]).copy()
districts = np.array(sorted(df["District"].unique()))

splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
_, heldout_idx = next(splitter.split(districts, groups=districts))
heldout_districts = districts[heldout_idx].tolist()

known = df[~df["District"].isin(heldout_districts)]
heldout = df[df["District"].isin(heldout_districts)]

train = known[known["Split"] == "train"].copy()
validation = known[known["Split"] == "validation"].copy()
test = heldout[heldout["Split"] == "test"].copy()

print("HELD-OUT DISTRICTS:", heldout_districts)
print("Training rows:", len(train))
print("Validation rows:", len(validation))
print("Geographic test rows:", len(test))
print("Held-out test stations:", test["Provisional_Station_ID"].nunique())

if min(len(train), len(validation), len(test)) == 0:
    raise ValueError("A geographic split is empty. Review district coverage.")

X_train, y_train = train[features], train["Target_GWL_m_bgl"]
X_val, y_val = validation[features], validation["Target_GWL_m_bgl"]
X_test, y_test = test[features], test["Target_GWL_m_bgl"]

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("model", LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=7,
        min_child_samples=20,
        reg_lambda=2,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        n_jobs=4,
        random_state=42,
        verbosity=-1
    ))
])

pipeline.fit(X_train, y_train)

def score(y, pred):
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "R2": r2_score(y, pred)
    }

rows = []
for name, y, pred in [
    ("Persistence", y_test, test["GWL_m_bgl"]),
    ("LightGBM", y_test, pipeline.predict(X_test))
]:
    rows.append({"Model": name, **score(y, pred)})

result = pd.DataFrame(rows)

out_dir = root / "results/metrics"
out_dir.mkdir(parents=True, exist_ok=True)
result.to_csv(out_dir / "geographic_holdout_results.csv", index=False)

(root / "reports/geographic_holdout_districts.json").write_text(
    pd.Series(heldout_districts).to_json(orient="values", indent=2),
    encoding="utf-8"
)

joblib.dump(pipeline, root / "models/trained/lightgbm_geographic_holdout.joblib")

print("\nGEOGRAPHIC HOLDOUT RESULTS")
print(result.round(4).to_string(index=False))
print("\nSaved geographic metrics:", out_dir / "geographic_holdout_results.csv")
print("Saved geographic model: models/trained/lightgbm_geographic_holdout.joblib")
