from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

source = ROOT / "reports/extreme_and_sudden_change_review.csv"
output = ROOT / "reports/anomaly_register.csv"

df = pd.read_csv(source)

df["Absolute_Change_m"] = pd.to_numeric(
    df["Change_From_Previous_m"], errors="coerce"
)

df["Priority"] = "REVIEW"

critical = (
    (df["GWL_m_bgl"] >= 150) |
    (df["Absolute_Change_m"] >= 100)
)

high = (
    (df["GWL_m_bgl"] > 100) |
    (df["Absolute_Change_m"] >= 50)
)

df.loc[high, "Priority"] = "HIGH"
df.loc[critical, "Priority"] = "CRITICAL"

df["Modelling_Decision"] = "RETAIN_PENDING_VERIFICATION"
df["Excluded_From_Primary_Data"] = False

df["Review_Note"] = (
    "Flagged for investigation; no measurement has been declared erroneous"
)

df.to_csv(output, index=False)

print("ANOMALY REGISTER CREATED")
print("Flagged observations:", len(df))
print("Affected provisional stations:",
      df["Provisional_Station_ID"].nunique())

print("\nPRIORITY COUNTS:")
print(df["Priority"].value_counts().to_string())

print("\nMODELLING DECISIONS:")
print(df["Modelling_Decision"].value_counts().to_string())

print("\nSAVED:")
print(output)

print("\nOriginal model-ready observations remain unchanged.")
