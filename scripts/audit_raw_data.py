from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "gwl_manual_quarterly_cgwb_tn_1991_2020.csv",
    "gwl_manual_quarterly_cgwb_tn_2021_2025.csv",
]

DATE_COL = "Data Acquisition Time"
TARGET_COL = "Groundwater Level Quarterly Manual (meter)"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(filename):
    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    result = {
        "filename": filename,
        "sha256": sha256_file(path),
        "file_size_bytes": path.stat().st_size,
        "rows": int(len(df)),
        "columns_count": int(len(df.columns)),
        "columns": list(df.columns),
        "data_types": {
            col: str(dtype) for col, dtype in df.dtypes.items()
        },
        "missing_values": {
            col: int(count)
            for col, count in df.isna().sum().items()
        },
        "complete_duplicate_rows": int(df.duplicated().sum()),
    }

    if DATE_COL in df.columns:
        dates = pd.to_datetime(
            df[DATE_COL],
            format="%d-%m-%Y %H:%M",
            errors="coerce"
        )

        valid_dates = dates.dropna()

        result["date_audit"] = {
            "invalid_dates": int(dates.isna().sum()),
            "minimum_date": (
                str(valid_dates.min()) if len(valid_dates) else None
            ),
            "maximum_date": (
                str(valid_dates.max()) if len(valid_dates) else None
            ),
        }

        if "Station" in df.columns:
            station_dates = pd.DataFrame({
                "station": df["Station"],
                "date": dates
            })

            result["duplicate_station_date_rows"] = int(
                station_dates.duplicated(
                    subset=["station", "date"],
                    keep=False
                ).sum()
            )

    for col in ["Station", "District", "Block", "Village"]:
        if col in df.columns:
            result[f"unique_{col.lower()}"] = int(
                df[col].nunique(dropna=True)
            )

    for col in ["Latitude", "Longitude"]:
        if col in df.columns:
            result[f"missing_{col.lower()}"] = int(df[col].isna().sum())

    if TARGET_COL in df.columns:
        values = pd.to_numeric(df[TARGET_COL], errors="coerce")
        result["groundwater_level"] = {
            "missing": int(values.isna().sum()),
            "minimum": float(values.min()) if values.notna().any() else None,
            "maximum": float(values.max()) if values.notna().any() else None,
            "mean": float(values.mean()) if values.notna().any() else None,
            "median": float(values.median()) if values.notna().any() else None,
            "negative_count": int((values < 0).sum()),
            "zero_count": int((values == 0).sum()),
            "positive_count": int((values > 0).sum()),
        }

    return result


def main():
    print("=" * 65)
    print("GROUNDWATER STRESS PREDICTION - RAW DATA AUDIT")
    print("=" * 65)

    results = []

    for filename in FILES:
        print(f"\nAuditing: {filename}")
        result = audit_dataset(filename)
        results.append(result)

        print(f"Rows: {result['rows']:,}")
        print(f"Columns: {result['columns_count']}")
        print(f"SHA256: {result['sha256']}")

        if "date_audit" in result:
            print("Date range:",
                  result["date_audit"]["minimum_date"],
                  "to",
                  result["date_audit"]["maximum_date"])

        print("Duplicate rows:",
              result["complete_duplicate_rows"])

        print("Missing values:")
        for col, count in result["missing_values"].items():
            if count:
                print(f"  {col}: {count:,}")

        if "groundwater_level" in result:
            print("Groundwater level summary:",
                  result["groundwater_level"])

    report = {
        "project": "Groundwater Stress Prediction",
        "audit_type": "Read-only raw dataset audit",
        "datasets": results,
    }

    report_path = REPORT_DIR / "raw_data_audit.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("AUDIT COMPLETED")
    print(f"Report saved: {report_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
