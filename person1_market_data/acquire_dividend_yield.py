
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from datetime import datetime
import pandas as pd
import numpy as np
import time
import io


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT / "data" / "raw" / "nifty50_daily.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_FILE = (
    OUTPUT_DIR / "nifty50_dividend_yield.csv"
)

BASE_URL = (
    "https://archives.nseindia.com/content/indices/"
    "ind_close_all_{}.csv"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/csv,*/*"
}


# ============================================================
# 2. DATE PARSING
# ============================================================

def parse_archive_date(value, expected_date=None):
    """
    Parse NSE archive dates.

    NSE date strings may have ambiguous day/month ordering.
    When expected_date is provided, accept only an interpretation
    matching the corresponding NIFTY trading date.
    """

    value = str(value).strip()

    supported_formats = [
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d"
    ]

    candidates = set()

    for date_format in supported_formats:

        try:
            parsed = datetime.strptime(
                value,
                date_format
            ).date()

            candidates.add(parsed)

        except ValueError:
            continue

    if not candidates:
        raise ValueError(
            f"Unsupported archive date format: {value!r}"
        )

    if expected_date is not None:

        if expected_date in candidates:
            return expected_date

        raise ValueError(
            f"Archive date mismatch. "
            f"Raw value: {value!r}; "
            f"Possible dates: {sorted(candidates)}; "
            f"Expected date: {expected_date}"
        )

    if len(candidates) != 1:
        raise ValueError(
            f"Ambiguous archive date: {value!r}. "
            f"Possible dates: {sorted(candidates)}"
        )

    return candidates.pop()


# ============================================================
# 3. DOWNLOAD NSE DAILY ARCHIVE
# ============================================================

def download_daily_file(trading_date):
    """
    Download the NSE daily index archive for one trading date.

    Retries temporary network and HTTP errors.
    Stops immediately if the archive returns HTTP 404.
    """

    date_code = trading_date.strftime("%d%m%Y")

    url = BASE_URL.format(date_code)

    request = Request(
        url,
        headers=HEADERS
    )

    delays = [5, 15, 30]

    for attempt in range(len(delays) + 1):

        try:

            with urlopen(
                request,
                timeout=60
            ) as response:

                if response.status != 200:
                    raise RuntimeError(
                        f"HTTP {response.status} for {date_code}"
                    )

                content = response.read()

            if len(content) < 5000:
                raise RuntimeError(
                    f"Unexpectedly small response for "
                    f"{date_code}: {len(content)} bytes"
                )

            return content

        except HTTPError as exc:

            if exc.code == 404:
                raise RuntimeError(
                    f"NSE archive missing for expected "
                    f"trading date {trading_date}"
                ) from exc

            if attempt == len(delays):
                raise

            print(
                f"HTTP error {exc.code}. "
                f"Retrying in {delays[attempt]} seconds..."
            )

        except (URLError, TimeoutError) as exc:

            if attempt == len(delays):
                raise

            print(
                f"Network error: {exc}. "
                f"Retrying in {delays[attempt]} seconds..."
            )

        time.sleep(delays[attempt])

    raise RuntimeError(
        f"Download failed for {trading_date}"
    )


# ============================================================
# 4. DIVIDEND YIELD VALIDATION
# ============================================================

def validate_dividend_yield(value, trading_date):
    """
    Validate the dividend yield.

    Known missing markers are retained as NaN.
    Unexpected non-numeric values cause an error.
    """

    if pd.isna(value):
        return np.nan

    text_value = str(value).strip()

    if text_value in ("", "-", "NA", "N/A", "nan"):
        return np.nan

    try:
        numeric_value = float(text_value)

    except ValueError as exc:
        raise ValueError(
            f"Non-numeric dividend yield on "
            f"{trading_date}: {value!r}"
        ) from exc

    if not np.isfinite(numeric_value):
        raise ValueError(
            f"Non-finite dividend yield on {trading_date}"
        )

    if numeric_value < 0:
        raise ValueError(
            f"Negative dividend yield on {trading_date}: "
            f"{numeric_value}"
        )

    return numeric_value


# ============================================================
# 5. MAIN ACQUISITION FUNCTION
# ============================================================

def acquire_dividend_yield():

    # --------------------------------------------------------
    # Check input and output files
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Required NIFTY dataset not found: {INPUT_FILE}"
        )

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}. "
            "Inspect it before replacing."
        )

    # --------------------------------------------------------
    # Load NIFTY trading dates
    # --------------------------------------------------------

    nifty = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date"]
    )

    if "Date" not in nifty.columns:
        raise ValueError(
            "NIFTY dataset does not contain a Date column."
        )

    if nifty["Date"].isna().any():
        raise ValueError(
            "Invalid dates found in NIFTY dataset."
        )

    if nifty["Date"].duplicated().any():
        raise ValueError(
            "Duplicate dates found in NIFTY dataset."
        )

    trading_dates = sorted(
        nifty["Date"].dt.date.unique()
    )

    if not trading_dates:
        raise ValueError(
            "No trading dates found in NIFTY dataset."
        )

    print("NIFTY 50 trading dates:", len(trading_dates))
    print("First date:", trading_dates[0])
    print("Last date:", trading_dates[-1])

    print("\nDownloading NSE daily index archives...\n")

    # --------------------------------------------------------
    # Download and validate each date
    # --------------------------------------------------------

    collected = []

    for number, trading_date in enumerate(
        trading_dates,
        start=1
    ):

        print(
            f"[{number}/{len(trading_dates)}] "
            f"{trading_date}"
        )

        raw = download_daily_file(trading_date)

        daily = pd.read_csv(
            io.BytesIO(raw)
        )

        required_columns = {
            "Index Name",
            "Index Date",
            "Div Yield"
        }

        if not required_columns.issubset(daily.columns):
            raise ValueError(
                f"Unexpected columns for {trading_date}: "
                f"{list(daily.columns)}"
            )

        # ----------------------------------------------------
        # Find the NIFTY 50 row
        # ----------------------------------------------------

        matching = daily[
            daily["Index Name"]
            .astype(str)
            .str.strip()
            .str.upper()
            == "NIFTY 50"
        ]

        if len(matching) != 1:
            raise ValueError(
                f"Expected exactly one NIFTY 50 row "
                f"for {trading_date}; found {len(matching)}"
            )

        row = matching.iloc[0].copy()

        # ----------------------------------------------------
        # Validate archive date
        # ----------------------------------------------------

        raw_source_date = row["Index Date"]

        try:

            source_date = parse_archive_date(
                raw_source_date,
                expected_date=trading_date
            )

        except ValueError:

            print("\nDATE MISMATCH DETAILS")
            print("---------------------")
            print("Requested date:", trading_date)
            print("Raw archive date:", repr(raw_source_date))
            print(
                "Archive URL:",
                BASE_URL.format(
                    trading_date.strftime("%d%m%Y")
                )
            )

            raise

        # Store normalized date in the output.
        row["Index Date"] = source_date

        # ----------------------------------------------------
        # Validate dividend yield
        # ----------------------------------------------------

        row["Div Yield"] = validate_dividend_yield(
            row["Div Yield"],
            trading_date
        )

        collected.append(row)

        # Avoid rapid requests.
        time.sleep(0.2)

    # ========================================================
    # 6. BUILD FINAL DATASET
    # ========================================================

    result = pd.DataFrame(collected)

    result["Index Date"] = pd.to_datetime(
        result["Index Date"]
    )

    result["Div Yield"] = pd.to_numeric(
        result["Div Yield"],
        errors="raise"
    )

    result = result.sort_values(
        "Index Date"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Final integrity checks
    # --------------------------------------------------------

    if len(result) != len(trading_dates):
        raise ValueError(
            "Observation count does not match "
            "NIFTY trading dates."
        )

    if result["Index Date"].duplicated().any():
        raise ValueError(
            "Duplicate dates detected in dividend dataset."
        )

    expected_dates = pd.to_datetime(
        trading_dates
    )

    if not result["Index Date"].reset_index(drop=True).equals(
        pd.Series(expected_dates, name="Index Date")
    ):
        raise ValueError(
            "Final dividend dataset dates do not exactly "
            "match the NIFTY trading dates."
        )

    # ========================================================
    # 7. SAVE DATASET ATOMICALLY
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_file = OUTPUT_FILE.with_suffix(".tmp")

    try:

        result.to_csv(
            temp_file,
            index=False
        )

        temp_file.replace(
            OUTPUT_FILE
        )

    finally:

        if temp_file.exists():
            temp_file.unlink()

    # ========================================================
    # 8. ACQUISITION REPORT
    # ========================================================

    print("\nDIVIDEND YIELD ACQUISITION SUCCESSFUL")
    print("------------------------------------")

    print("Source: NSE daily index archive")
    print("Observations:", len(result))

    print(
        "First date:",
        result["Index Date"].min().date()
    )

    print(
        "Last date:",
        result["Index Date"].max().date()
    )

    print(
        "Missing dividend yields:",
        result["Div Yield"].isna().sum()
    )

    print(
        "Duplicate dates:",
        result["Index Date"].duplicated().sum()
    )

    print("Columns:", list(result.columns))
    print("Saved:", OUTPUT_FILE)


# ============================================================
# 9. EXECUTION
# ============================================================

if __name__ == "__main__":
    acquire_dividend_yield()
