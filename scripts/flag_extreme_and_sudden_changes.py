from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

source = ROOT / "data/processed/observations_model_ready.csv"
report_dir = ROOT / "reports"

df = pd.read_csv(source)
df["Parsed_Date"] = pd.to_datetime(df["Parsed_Date"], errors="coerce")

df = df.sort_values(
    ["Provisional_Station_ID", "Parsed_Date"]
).copy()

grouped = df.groupby("Provisional_Station_ID", sort=False)

df["Previous_Date"] = grouped["Parsed_Date"].shift(1)
df["Previous_GWL_m"] = grouped["GWL_m_bgl"].shift(1)

df["Days_Since_Previous"] = (
    df["Parsed_Date"] - df["Previous_Date"]
).dt.days

df["Change_From_Previous_m"] = (
    df["GWL_m_bgl"] - df["Previous_GWL_m"]
)

df["Absolute_Change_m"] = df["Change_From_Previous_m"].abs()

df["Flag_Above_100m"] = df["GWL_m_bgl"] > 100

df["Flag_Sudden_Change"] = (
    (df["Absolute_Change_m"] >= 50) &
    (df["Days_Since_Previous"] <= 200) &
    (df["Days_Since_Previous"] >= 0)
)

flagged = df[
    df["Flag_Above_100m"] | df["Flag_Sudden_Change"]
].copy()

flagged["Investigation_Reason"] = flagged.apply(
    lambda r: "; ".join(
        reason for condition, reason in [
            (r["Flag_Above_100m"], "READING_ABOVE_100M"),
            (r["Flag_Sudden_Change"], "CHANGE_OF_50M_OR_MORE_WITHIN_200_DAYS")
        ] if condition
    ),
    axis=1
)

columns = [
    "Provisional_Station_ID",
    "Station",
    "District",
    "Parsed_Date",
    "GWL_m_bgl",
    "Previous_Date",
    "Previous_GWL_m",
    "Days_Since_Previous",
    "Change_From_Previous_m",
    "Investigation_Reason",
    "Source_File",
    "Source_Row_Number"
]

flagged[columns].to_csv(
    report_dir / "extreme_and_sudden_change_review.csv",
    index=False
)

print("OBSERVATIONS FLAGGED:", len(flagged))
print("READINGS ABOVE 100 M:", int(flagged["Flag_Above_100m"].sum()))
print("SUDDEN CHANGES:", int(flagged["Flag_Sudden_Change"].sum()))

print("\nFLAGGED OBSERVATIONS:")
print(flagged[columns].to_string(index=False))

print("\nSAVED:")
print(report_dir / "extreme_and_sudden_change_review.csv")
print("\nNo observations were removed or modified.")
