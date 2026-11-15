# Person A — Volatility Sensitivity
As-of date: 2026-11-06
Underlying: NIFTY 50
Expiry: 2026-12-08
Primary strike: 25150 CE
Spot: 25030.35
r: 0.05854071082141279
q: 0.0141
T: 32/365

## Historical volatility-window sensitivity

| Window | Sigma | BS call price (INR) | Change from 90-day (INR) |
|---|---:|---:|---:|
| 90-day | 0.1296339024 | 372.1436 | 0.0000 |
| 180-day | 0.1573804100 | 454.0798 | 81.9362 |
| 250-day | 0.1438012600 | 413.9803 | 41.8366 |

## Local sensitivity: plus/minus one percentage point

| Sigma | BS call price (INR) | Change from baseline (INR) |
|---|---:|---:|
| 0.1196339024 | 342.6137 | -29.5299 |
| 0.1296339024 | 372.1436 | 0.0000 |
| 0.1396339024 | 401.6739 | 29.5303 |

## Interpretation

The option price is materially sensitive to the historical volatility
estimation window. The 180-day estimate produces the highest price,
followed by the 250-day and 90-day estimates.

A one-percentage-point change around the primary 90-day volatility
changes the ATM call value by approximately INR 29.53.

All other model inputs are held fixed. These are analytical model
sensitivities, not market-price forecasts.
