
from pathlib import Path
from urllib.request import Request, urlopen
from datetime import datetime, timezone
import csv
import io

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "rbi_91day_tbill_auctions.csv"

# Historical RBI-sourced 91-day T-Bill auction data
SOURCE_URL = (
    "https://measure.thisindianlife.today/"
    "data/rates-tbill-91.csv"
)

def acquire_risk_free_data():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Protect existing raw datasets
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"File already exists: {OUTPUT_FILE}. "
            "Inspect it before replacing."
        )

    print("Downloading 91-day Treasury Bill auction data...")

    request = Request(
        SOURCE_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(
                f"Download failed: HTTP {response.status}"
            )

        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        raw_data = response.read()

    if not raw_data:
        raise RuntimeError("Downloaded file is empty.")

    # Reject HTML error pages disguised as downloads
    if b"<html" in raw_data[:1000].lower():
        raise RuntimeError(
            "Received HTML instead of CSV data."
        )

    # Validate CSV structure without modifying source values
    try:
        text_data = raw_data.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text_data))
        rows = list(reader)
    except Exception as exc:
        raise RuntimeError(
            f"Could not parse downloaded CSV: {exc}"
        )

    if len(rows) < 2:
        raise RuntimeError(
            "CSV contains no usable data rows."
        )

    headers = rows[0]

    if not headers or len(headers) < 2:
        raise RuntimeError(
            "Unexpected CSV structure."
        )

    # Atomic write; preserve the original CSV contents
    temp_file = OUTPUT_FILE.with_suffix(".tmp")
    temp_file.write_bytes(raw_data)
    temp_file.replace(OUTPUT_FILE)

    print("\nRISK-FREE DATA ACQUISITION SUCCESSFUL")
    print("------------------------------------")
    print("Instrument: 91-day Government of India T-Bill")
    print("Source:", SOURCE_URL)
    print("Content type:", content_type)
    print("CSV columns:", headers)
    print("Data rows:", len(rows) - 1)
    print("Downloaded at:", datetime.now(
        timezone.utc
    ).isoformat())
    print("File size:", OUTPUT_FILE.stat().st_size, "bytes")
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    acquire_risk_free_data()
