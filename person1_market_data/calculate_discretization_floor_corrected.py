from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FINAL = DATA / "final" / "person1"

with open(FINAL / "person1_gbm_inputs_20261106.json", encoding="utf-8") as f:
    inputs = json.load(f)

strike_data = pd.read_csv(FINAL / "person1_selected_strikes_20261106.csv")
K = float(strike_data.loc[
    strike_data["selection"] == "ATM", "selected_strike"
].iloc[0])

S0 = float(inputs["spot_price_S0"])
r = float(inputs["continuous_risk_free_rate_r"])
q = float(inputs["dividend_yield_decimal_q"])
sigma = float(inputs["primary_sigma"])
T = float(inputs["time_to_expiry_years"])

k = 4
mu = math.log(S0) + (r - q - 0.5 * sigma**2) * T
v = sigma * math.sqrt(T)

S_min = math.exp(mu - k * v)
S_max = math.exp(mu + k * v)

def cdf_price(x):
    if x <= 0:
        return 0.0
    return float(ndtr((math.log(x) - mu) / v))

mass = cdf_price(S_max) - cdf_price(S_min)

def full_bs_call(strike):
    d1 = (
        math.log(S0 / strike)
        + (r - q + 0.5 * sigma**2) * T
    ) / v
    d2 = d1 - v
    return (
        S0 * math.exp(-q*T) * ndtr(d1)
        - strike * math.exp(-r*T) * ndtr(d2)
    )

def truncated_conditional_call(strike):
    # Integrate the call payoff only over [S_min, S_max],
    # then condition on being inside that range.
    lower = max(strike, S_min)

    if lower >= S_max:
        return 0.0

    probability = cdf_price(S_max) - cdf_price(lower)

    first_moment = (
        math.exp(mu + 0.5*v*v)
        * (
            ndtr((math.log(S_max)-mu-v*v)/v)
            - ndtr((math.log(lower)-mu-v*v)/v)
        )
    )

    expected_payoff = (first_moment - strike*probability) / mass
    return math.exp(-r*T) * expected_payoff

rows = []

for n in range(3, 21):
    N = 2**n
    S = np.linspace(S_min, S_max, N)

    index = int(np.argmin(np.abs(S-K)))
    K_grid = float(S[index])

    boundaries = np.empty(N+1)
    boundaries[0] = S_min
    boundaries[-1] = S_max
    boundaries[1:-1] = (S[:-1]+S[1:])/2

    probabilities = np.diff(
        np.array([cdf_price(x) for x in boundaries])
    ) / mass

    V_grid = math.exp(-r*T) * float(
        np.dot(probabilities, np.maximum(S-K_grid, 0))
    )

    V_truncated = truncated_conditional_call(K_grid)
    V_full = full_bs_call(K_grid)

    rows.append({
        "n": n,
        "number_of_points": N,
        "grid_spacing": S[1]-S[0],
        "original_strike": K,
        "snapped_strike": K_grid,
        "V_grid": V_grid,
        "V_continuous_truncated": V_truncated,
        "discretization_floor": abs(V_grid-V_truncated),
        "V_BS_full_distribution": V_full,
        "truncation_conditioning_difference": abs(V_truncated-V_full),
        "total_difference_from_full_BS": abs(V_grid-V_full),
        "probability_sum": probabilities.sum()
    })

result = pd.DataFrame(rows)
output = FINAL / "person1_discretization_floor_corrected_20261106.csv"
result.to_csv(output, index=False)

print("CORRECTED FLOOR CALCULATION COMPLETE")
print("Saved:", output)
print(result.to_string(index=False, float_format=lambda x: f"{x:.10f}"))
print("\nConvergence:")
print("First discretization floor:", result.iloc[0]["discretization_floor"])
print("Last discretization floor:", result.iloc[-1]["discretization_floor"])
