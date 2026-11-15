
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

import shutil

MIN_FREE_SPACE_GB = 10

def check_disk_space(path):
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024 ** 3)

    print(f"Available disk space: {free_gb:.2f} GB")

    if free_gb < MIN_FREE_SPACE_GB:
        raise RuntimeError(
            "Insufficient disk space. "
            "Download cancelled."
        )

    return free_gb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "nifty50_daily.csv"

TICKER = "^NSEI"


def acquire_nifty_data():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    check_disk_space(PROJECT_ROOT)
    # Never silently overwrite an existing raw dataset.
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Dataset already exists: {OUTPUT_FILE}. "
            "Inspect it before deciding whether to replace it."
        )

    print("Downloading five years of NIFTY 50 daily data...")

    df = yf.download(
        TICKER,
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise RuntimeError("Yahoo Finance returned no data.")

    # Handle Yahoo Finance MultiIndex columns.
    if isinstance(df.columns, pd.MultiIndex):
        ticker_level = None

        for level in range(df.columns.nlevels):
            if TICKER in df.columns.get_level_values(level):
                ticker_level = level
                break

        if ticker_level is None:
            raise ValueError("Could not identify NIFTY ticker columns.")

        df = df.xs(TICKER, axis=1, level=ticker_level)

    df.index = pd.to_datetime(df.index)

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df.index.name = "Date"
    df = df.sort_index()

    required = ["Open", "High", "Low", "Close", "Volume"]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required].copy()

    # Validate numerical values.
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains NaN or infinite values.")

    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Non-positive price detected.")

    if (df["Volume"] < 0).any():
        raise ValueError("Negative volume detected.")

    if (df["High"] < df[["Open", "Close"]].max(axis=1)).any():
        raise ValueError("Invalid High price detected.")

    if (df["Low"] > df[["Open", "Close"]].min(axis=1)).any():
        raise ValueError("Invalid Low price detected.")

    if (df["High"] < df["Low"]).any():
        raise ValueError("High is below Low.")

    if df.index.has_duplicates:
        raise ValueError("Duplicate trading dates detected.")

    if len(df) < 900:
        raise ValueError(
            f"Unexpectedly few observations: {len(df)}"
        )

    # Atomic write: do not leave a partially written dataset.
    temp_file = OUTPUT_FILE.with_suffix(".tmp")
    df.to_csv(temp_file, index=True)
    temp_file.replace(OUTPUT_FILE)

    size_kb = OUTPUT_FILE.stat().st_size / 1024

    print("\nDATA ACQUISITION SUCCESSFUL")
    print("---------------------------")
    print("Ticker:", TICKER)
    print("Observations:", len(df))
    print("First date:", df.index.min().date())
    print("Last date:", df.index.max().date())
    print("Columns:", list(df.columns))
    print(f"File size: {size_kb:.2f} KB")
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    acquire_nifty_data()
