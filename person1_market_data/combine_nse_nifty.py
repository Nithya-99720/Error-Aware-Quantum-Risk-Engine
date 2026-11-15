
from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import json

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / "data" / "raw" / "nse_nifty_batches"
OLD_FILE = ROOT / "data" / "raw" / "nifty50_daily.csv"

OUTPUT = ROOT / "data" / "processed" / "nifty50_nse_official_clean.csv"
REPORT = ROOT / "data" / "processed" / "nifty50_nse_audit.json"

EXPECTED_COLUMNS = [
    "Index Name", "Date", "Open", "High", "Low", "Close"
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    files = sorted(FOLDER.glob("*.csv"))

    if len(files) != 5:
        raise ValueError(f"Expected 5 NSE CSV files, found {len(files)}")

    frames = []

    for file in files:
        df = pd.read_csv(file)

        if list(df.columns) != EXPECTED_COLUMNS:
            raise ValueError(
                f"Unexpected columns in {file.name}: {list(df.columns)}"
            )

        df["source_file"] = file.name
        frames.append(df)

        print(f"{file.name}: {len(df)} rows")

    combined = pd.concat(frames, ignore_index=True)

    # Validate index name.
    if set(combined["Index Name"].dropna().unique()) != {"NIFTY 50"}:
        raise ValueError("Unexpected or missing index names.")

    # Parse dates and numeric columns.
    combined["Date"] = pd.to_datetime(
        combined["Date"], format="%d %b %Y", errors="coerce"
    )

    if combined["Date"].isna().any():
        raise ValueError("Invalid dates found in NSE files.")

    for col in ["Open", "High", "Low", "Close"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    if combined[["Open", "High", "Low", "Close"]].isna().any().any():
        raise ValueError("Missing or non-numeric OHLC values found.")

    # Check duplicate dates before creating the clean dataset.
    duplicates = combined[
        combined.duplicated(subset=["Date"], keep=False)
    ].sort_values("Date")

    if not duplicates.empty:
        print("\nDUPLICATE DATES:")
        print(duplicates.to_string(index=False))
        raise ValueError(
            "Duplicate dates found. Resolve them before combining."
        )

    # Basic OHLC consistency checks.
    invalid_ohlc = combined[
        (combined["High"] < combined["Low"])
        | (combined["High"] < combined["Open"])
        | (combined["High"] < combined["Close"])
        | (combined["Low"] > combined["Open"])
        | (combined["Low"] > combined["Close"])
    ]

    if not invalid_ohlc.empty:
        raise ValueError("Invalid OHLC relationships found.")

    # Sort chronologically.
    combined = combined.sort_values("Date").reset_index(drop=True)

    # Standardized analytical file.
    clean = combined.rename(columns={
        "Index Name": "index_name",
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close"
    })

    clean["date"] = clean["date"].dt.strftime("%Y-%m-%d")
    clean = clean[
        ["date", "index_name", "open", "high", "low", "close"]
    ]

    # Compare dates and closing prices with the previous dataset.
    old = pd.read_csv(OLD_FILE)
    old["Date"] = pd.to_datetime(old["Date"], errors="coerce")
    old["Close"] = pd.to_numeric(old["Close"], errors="coerce")
    old = old.dropna(subset=["Date", "Close"])

    official_dates = set(pd.to_datetime(clean["date"]))
    old_dates = set(old["Date"])

    only_official = sorted(official_dates - old_dates)
    only_old = sorted(old_dates - official_dates)

    comparison = clean.copy()
    comparison["date"] = pd.to_datetime(comparison["date"])

    comparison = comparison.merge(
        old[["Date", "Close"]],
        left_on="date",
        right_on="Date",
        how="inner"
    )

    comparison["close_difference"] = (
        comparison["close"] - comparison["Close"]
    )

    price_mismatches = comparison[
        comparison["close_difference"].abs() > 0.01
    ]

    # Save outputs.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUTPUT, index=False)

    report = {
        "source_files": [
            {
                "name": f.name,
                "rows": len(pd.read_csv(f)),
                "sha256": sha256(f)
            }
            for f in files
        ],
        "combined_rows": len(clean),
        "unique_dates": int(clean["date"].nunique()),
        "duplicate_dates": int(clean["date"].duplicated().sum()),
        "start_date": clean["date"].min(),
        "end_date": clean["date"].max(),
        "missing_ohlc_values": int(
            clean[["open", "high", "low", "close"]].isna().sum().sum()
        ),
        "official_dates_not_in_old_file": [
            d.strftime("%Y-%m-%d") for d in only_official
        ],
        "old_file_dates_not_in_official": [
            d.strftime("%Y-%m-%d") for d in only_old
        ],
        "common_dates": len(comparison),
        "close_price_mismatches_over_0_01": len(price_mismatches),
        "maximum_absolute_close_difference": (
            float(comparison["close_difference"].abs().max())
            if len(comparison) else None
        ),
        "output_file": str(OUTPUT),
        "output_sha256": sha256(OUTPUT)
    }

    REPORT.write_text(json.dumps(report, indent=4), encoding="utf-8")

    print("\nNSE COMBINATION COMPLETE")
    print("Combined rows:", len(clean))
    print("Unique dates:", clean["date"].nunique())
    print("Duplicate dates:", clean["date"].duplicated().sum())
    print("Date range:", clean["date"].min(), "to", clean["date"].max())
    print("Dates only in official file:", len(only_official))
    print("Dates only in old file:", len(only_old))
    print("Common dates:", len(comparison))
    print("Close mismatches > ₹0.01:", len(price_mismatches))
    print("Maximum absolute close difference:",
          report["maximum_absolute_close_difference"])
    print("Clean file:", OUTPUT)
    print("Audit report:", REPORT)


if __name__ == "__main__":
    main()
