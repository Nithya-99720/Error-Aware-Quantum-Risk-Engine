## RBI 91-day Treasury Bill Auction Data

File: data/raw/rbi_91day_tbill_auctions.csv

Underlying data authority: Reserve Bank of India (RBI), Database on Indian Economy (DBIE), auction-wise 91-day Government of India Treasury Bill data.

Original download provider and URL: Not recoverable from available download records.

Independent reference:
https://measure.thisindianlife.today/data/rates-tbill-91/

Official RBI reference:
https://www.rbi.org.in/Scripts/PublicationsView.aspx?id=24043

Provenance limitation: The original third-party download URL and download date have not been independently verified. The independent reference is provided for corroboration, not as proof of the original download source.

## Explanation of five missing 91-day cut-off yields

The following dates have no reported 91-day cut-off yield:

| Auction date | Official RBI finding |
|---|---|
| 2016-01-27 | Auction held; no competitive bids accepted; no cut-off yield |
| 2016-02-24 | Auction held; no competitive bids accepted; no cut-off yield |
| 2023-03-29 | 164 competitive bids received; zero accepted |
| 2025-02-20 | Competitive bids received; no 91-day bids accepted |
| 2026-03-25 | 79 competitive bids received; zero accepted |

These are auction outcomes, not evidence of cancelled auctions or missing observations caused by data processing.

Treatment:
- Preserve the missing values as missing.
- Do not interpolate or forward-fill them.
- Use the latest valid available yield for the model's as-of date, with its observation date recorded.

Official RBI references:
- 2016 auction results: https://www.rbi.org.in/Scripts/BS_ViewBulletin.aspx?Id=16195
- 2023 auction results: https://www.rbi.org.in/Scripts/BS_ViewBulletin.aspx?Id=21892
- 2025 auction result: https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=59801
- 2026 auction result: https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?prid=62440

Checked: 2026-11-14
