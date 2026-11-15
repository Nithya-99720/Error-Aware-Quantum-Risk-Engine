from pathlib import Path
from datetime import date
import argparse
import json
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AS_OF = "2026-11-06"
EXPIRY = "2026-12-08"

SPOT_FILE = ROOT / "data/processed/nifty50_nse_official_clean.csv"
DIVIDEND_FILE = ROOT / "data/raw/nifty50_dividend_yield.csv"
RBI_FILE = ROOT / "data/processed/rbi_91day_tbill_yield_clean.csv"
OUTPUT_FILE = ROOT / "data/final/person1/person1_gbm_inputs_20261106.json"


def build_inputs():
    as_of = pd.Timestamp(AS_OF)
    expiry = pd.Timestamp(EXPIRY)

    # Spot price from official NSE closing-price history.
    spot = pd.read_csv(SPOT_FILE)
    spot["date"] = pd.to_datetime(spot["date"])
    spot = spot[spot["date"] <= as_of].sort_values("date")
    spot_row = spot.iloc[-1]

    if spot_row["date"] != as_of:
        raise ValueError("NSE spot data does not contain the as-of date.")

    s0 = float(spot_row["close"])

    # Historical volatility: daily log returns, sample standard deviation.
    log_returns = np.log(spot["close"].astype(float)).diff().dropna()
    volatility = {
        str(n): float(log_returns.tail(n).std(ddof=1) * np.sqrt(252))
        for n in (90, 180, 250)
    }

    if len(log_returns) < 250:
        raise ValueError("Insufficient history for 250-return volatility.")

    # Latest dividend yield on or before the as-of date.
    dividends = pd.read_csv(DIVIDEND_FILE)
    dividends["Index Date"] = pd.to_datetime(dividends["Index Date"])
    dividends = dividends[dividends["Index Date"] <= as_of].sort_values("Index Date")
    dividend_row = dividends.iloc[-1]

    if dividend_row["Index Date"] != as_of:
        raise ValueError("Dividend data does not contain the as-of date.")

    dividend_percent = float(dividend_row["Div Yield"])
    q = dividend_percent / 100.0

    # Latest reported RBI 91-day cut-off yield on or before the as-of date.
    rbi = pd.read_csv(RBI_FILE)
    rbi["date"] = pd.to_datetime(rbi["date"])
    rbi["cut_off_yield_percent"] = pd.to_numeric(
        rbi["cut_off_yield_percent"], errors="coerce"
    )
    rbi = rbi[
        (rbi["date"] <= as_of)
        & rbi["cut_off_yield_percent"].notna()
    ].sort_values("date")

    rbi_row = rbi.iloc[-1]
    yield_percent = float(rbi_row["cut_off_yield_percent"])
    y = yield_percent / 100.0

    # Convert 91-day simple annualized yield to continuous rate.
    r = (365.0 / 91.0) * math.log(1.0 + y * 91.0 / 365.0)

    calendar_days = (expiry - as_of).days
    if calendar_days <= 0:
        raise ValueError("Expiry must be after the as-of date.")

    t = calendar_days / 365.0
    forward = s0 * math.exp((r - q) * t)

    return {
        "as_of_date": AS_OF,
        "expiry_date": EXPIRY,
        "spot_price_S0": s0,
        "dividend_yield_percent": dividend_percent,
        "dividend_yield_decimal_q": q,
        "rbi_observation_date": rbi_row["date"].strftime("%Y-%m-%d"),
        "rbi_91_day_cutoff_yield_percent": yield_percent,
        "continuous_risk_free_rate_r": r,
        "volatility_annualized": volatility,
        "primary_volatility_window_days": 90,
        "primary_sigma": volatility["90"],
        "calendar_days_to_expiry": calendar_days,
        "time_to_expiry_years": t,
        "forward_price_F": forward,
        "volatility_method": (
            "Daily log returns, sample standard deviation (ddof=1), "
            "annualized by sqrt(252)"
        ),
        "risk_free_rate_method": "r = (365/91) * ln(1 + y*91/365)",
        "model_note": (
            "Historical volatility used as the volatility input "
            "in a risk-neutral GBM."
        ),
    }


def compare_values(generated, existing, path=""):
    differences = []

    for key, value in generated.items():
        current_path = f"{path}.{key}" if path else key

        if isinstance(value, dict):
            differences.extend(
                compare_values(value, existing.get(key, {}), current_path)
            )
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            old = existing.get(key)
            if old is None or not math.isclose(
                float(value), float(old), rel_tol=1e-10, abs_tol=1e-10
            ):
                differences.append(
                    f"{current_path}: generated={value}, existing={old}"
                )
        elif value != existing.get(key):
            differences.append(
                f"{current_path}: generated={value}, existing={existing.get(key)}"
            )

    return differences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="Replace the existing JSON only if all values match.",
    )
    args = parser.parse_args()

    generated = build_inputs()

    if not OUTPUT_FILE.exists():
        raise FileNotFoundError(f"Existing reference JSON not found: {OUTPUT_FILE}")

    existing = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
    differences = compare_values(generated, existing)

    print("GENERATED GBM INPUTS")
    print(json.dumps(generated, indent=4))

    if differences:
        print("\nVALIDATION FAILED. Existing JSON was not changed.")
        for difference in differences:
            print("-", difference)
        raise SystemExit(1)

    print("\nVALIDATION PASSED: all generated values match existing JSON.")

    if args.write:
        OUTPUT_FILE.write_text(
            json.dumps(generated, indent=4) + "\n",
            encoding="utf-8",
        )
        print("Updated:", OUTPUT_FILE)
    else:
        print("No file was overwritten. Use --write after reviewing the results.")


if __name__ == "__main__":
    main()
