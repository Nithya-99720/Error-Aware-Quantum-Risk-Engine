
from pathlib import Path
import hashlib
import json
import pandas as pd

# Project root: parent of person1_market_data
ROOT = Path(__file__).resolve().parent.parent

INPUT = (
    ROOT / "data" / "raw" / "fo_bhavcopy_extracted"
    / "BhavCopy_NSE_FO_0_0_0_20261106_F_0000.csv"
)

OUTPUT = ROOT / "data" / "processed" / "options_clean_20261106.csv"
REPORT = ROOT / "data" / "processed" / "options_quality_20261106.json"

AS_OF_DATE = pd.Timestamp("2026-11-06")
MIN_DAYS = 25
MAX_DAYS = 35
TARGET_DAYS = 30
MONEYNESS = 0.05
MIN_VOLUME = 1
MIN_OI = 1

REQUIRED = [
    "TradDt", "FinInstrmTp", "TckrSymb", "XpryDt",
    "StrkPric", "OptnTp", "ClsPric", "SttlmPric",
    "UndrlygPric", "OpnIntrst", "TtlTradgVol"
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    df = pd.read_csv(INPUT, low_memory=False)

    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse dates and numeric fields.
    df["trade_date"] = pd.to_datetime(df["TradDt"], errors="coerce")
    df["expiry"] = pd.to_datetime(df["XpryDt"], errors="coerce")

    numeric_columns = [
        "StrkPric", "ClsPric", "SttlmPric",
        "UndrlygPric", "OpnIntrst", "TtlTradgVol"
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Select the requested trading day's NIFTY index options.
    d = df[
        (df["trade_date"] == AS_OF_DATE)
        & (df["TckrSymb"] == "NIFTY")
        & (df["FinInstrmTp"] == "IDO")
        & (df["OptnTp"].isin(["CE", "PE"]))
    ].copy()

    if d.empty:
        raise ValueError("No NIFTY IDO option records found.")

    d["days_to_expiry"] = (d["expiry"] - AS_OF_DATE).dt.days

    # Select the expiry closest to 30 days within the required window.
    eligible = d[
        d["days_to_expiry"].between(MIN_DAYS, MAX_DAYS)
    ].copy()

    if eligible.empty:
        raise ValueError("No expiry found within the 25–35 day window.")

    expiry_counts = (
        eligible.groupby("expiry")["StrkPric"].count()
    )
    selected_expiry = min(
        expiry_counts.index,
        key=lambda x: abs((x - AS_OF_DATE).days - TARGET_DAYS)
    )

    selected = eligible[eligible["expiry"] == selected_expiry].copy()
    days = (selected_expiry - AS_OF_DATE).days

    # Underlying price must be available.
    underlying_values = selected["UndrlygPric"].dropna()
    if underlying_values.empty:
        raise ValueError("Underlying price is missing.")

    underlying = float(underlying_values.median())

    # Verify underlying-price consistency.
    underlying_min = float(underlying_values.min())
    underlying_max = float(underlying_values.max())

    # Apply the documented initial liquidity and moneyness screen.
    candidates = selected[
        (selected["StrkPric"] > 0)
        & (selected["TtlTradgVol"] >= MIN_VOLUME)
        & (selected["OpnIntrst"] >= MIN_OI)
        & (selected["ClsPric"] > 0)
        & (selected["SttlmPric"] > 0)
        & (
            (selected["StrkPric"] - underlying).abs()
            / underlying <= MONEYNESS
        )
    ].copy()

    if candidates.empty:
        raise ValueError("No contracts remain after filtering.")

    # Duplicate contract check: do not silently aggregate duplicates.
    duplicate_mask = candidates.duplicated(
        subset=["expiry", "StrkPric", "OptnTp"],
        keep=False
    )
    duplicate_count = int(duplicate_mask.sum())

    if duplicate_count:
        raise ValueError(
            "Duplicate expiry/strike/option-type contracts found:\n"
            + candidates.loc[
                duplicate_mask,
                ["expiry", "StrkPric", "OptnTp"]
            ].to_string(index=False)
        )

    # Clean long-format dataset: one row per option contract.
    clean = pd.DataFrame({
        "trade_date": AS_OF_DATE.date().isoformat(),
        "expiry": selected_expiry.date().isoformat(),
        "days_to_expiry": days,
        "T_years": days / 365.0,
        "underlying_price": underlying,
        "strike": candidates["StrkPric"],
        "option_type": candidates["OptnTp"],
        "close_price": candidates["ClsPric"],
        "settlement_price": candidates["SttlmPric"],
        "close_minus_settlement": (
            candidates["ClsPric"] - candidates["SttlmPric"]
        ),
        "open_interest": candidates["OpnIntrst"],
        "volume": candidates["TtlTradgVol"],
    })

    clean = clean.sort_values(
        ["strike", "option_type"]
    ).reset_index(drop=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUTPUT, index=False)

    # Quality report.
    report = {
        "input_file": str(INPUT),
        "input_sha256": sha256(INPUT),
        "output_file": str(OUTPUT),
        "output_sha256": sha256(OUTPUT),
        "as_of_date": AS_OF_DATE.date().isoformat(),
        "selected_expiry": selected_expiry.date().isoformat(),
        "days_to_expiry": days,
        "T_years": days / 365.0,
        "underlying_price_median": underlying,
        "underlying_price_min": underlying_min,
        "underlying_price_max": underlying_max,
        "filters": {
            "instrument_type": "IDO",
            "symbol": "NIFTY",
            "expiry_days_min": MIN_DAYS,
            "expiry_days_max": MAX_DAYS,
            "target_expiry_days": TARGET_DAYS,
            "absolute_moneyness_max": MONEYNESS,
            "minimum_volume": MIN_VOLUME,
            "minimum_open_interest": MIN_OI,
            "positive_close_required": True,
            "positive_settlement_required": True
        },
        "candidate_contracts": int(len(clean)),
        "call_contracts": int((clean["option_type"] == "CE").sum()),
        "put_contracts": int((clean["option_type"] == "PE").sum()),
        "unique_strikes": int(clean["strike"].nunique()),
        "contracts_with_both_prices": int(
            (clean["close_price"].notna()
             & clean["settlement_price"].notna()).sum()
        ),
        "duplicate_contract_groups": duplicate_count,
        "source_page": "https://www.nseindia.com/all-reports",
        "download_date": "2026-11-09",
        "notes": [
            "Original ZIP and extracted CSV are preserved.",
            "Close and settlement prices are retained separately.",
            "No bid/ask prices are available in this bhavcopy.",
            "Positive volume and open interest are an initial screen, "
            "not a guarantee of strong liquidity.",
            "Source page is recorded; exact archive download URL "
            "should be added to the project audit notes."
        ]
    }

    REPORT.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("OPTIONS PROCESSING COMPLETE")
    print("Selected expiry:", selected_expiry.date())
    print("Days to expiry:", days)
    print("Underlying:", underlying)
    print("Final contracts:", len(clean))
    print("Calls:", (clean["option_type"] == "CE").sum())
    print("Puts:", (clean["option_type"] == "PE").sum())
    print("Unique strikes:", clean["strike"].nunique())
    print("Duplicate groups:", duplicate_count)
    print("Clean dataset:", OUTPUT)
    print("Quality report:", REPORT)


if __name__ == "__main__":
    main()
