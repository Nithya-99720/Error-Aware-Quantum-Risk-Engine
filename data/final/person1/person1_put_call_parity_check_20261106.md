# Put–Call Parity Check — 6 November 2026

## Matched option contracts

- Underlying: NIFTY 50
- As-of date: 2026-11-06
- Expiry: 2026-12-08
- Days to expiry: 32
- Strike: 25150
- Call close: INR 369.95
- Put close: INR 457.55
- Call and put settlement prices also equal their respective close prices.

## Model inputs

- Spot S0: INR 25030.35
- Continuously compounded risk-free rate r: 0.05854071082141279
- Dividend yield q: 0.0141
- T: 32/365

## Calculation

GBM forward:

F_GBM = S0 * exp((r-q)*T)
      = INR 25128.0628

Put-call parity:

C - P = exp(-r*T) * (F-K)

F_parity = K + (C-P)*exp(r*T)
         = INR 25061.9493

Difference:

F_parity - F_GBM = -INR 66.1135
Relative difference = approximately -0.2631%

Discounted parity residual:

(C-P) - (S0*exp(-q*T) - K*exp(-r*T))
= -INR 65.7751

## Interpretation and limitations

The market-implied forward from the matched 25150 CE/PE closing prices
is INR 66.11 below the GBM forward.

This is a diagnostic comparison, not proof of arbitrage or model failure.
The bhavcopy does not provide bid/ask quotes. Transaction costs, liquidity,
market microstructure and asynchronous price formation cannot be assessed
from this dataset.

The market and model forwards are therefore reported separately.
No adjustment has been made to force agreement.

Option price convention: closing price.
Source: NSE F&O UDiFF Common Bhavcopy Final, trade date 2026-11-06.
