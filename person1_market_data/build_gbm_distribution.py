from pathlib import Path
import json
import math
import numpy as np
import pandas as pd


ROOT = Path(".")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA_FINAL = DATA / "final" / "person1"
SCRIPT_DIR = ROOT / "person1_market_data"
SCRIPT_DIR.mkdir(exist_ok=True)

# Load previously recorded inputs
with open(DATA_FINAL / "person1_gbm_inputs_20261106.json", encoding="utf-8") as f:
    inputs = json.load(f)

strikes = pd.read_csv(DATA_FINAL / "person1_selected_strikes_20261106.csv")
atm = strikes[strikes["selection"] == "ATM"]

if atm.empty:
    raise ValueError("ATM strike not found in selected-strikes file")

K = float(atm.iloc[0]["selected_strike"])

S0 = float(inputs["spot_price_S0"])
r = float(inputs["continuous_risk_free_rate_r"])
q = float(inputs["dividend_yield_decimal_q"])
sigma = float(inputs["primary_sigma"])
T = float(inputs["time_to_expiry_years"])
F = float(inputs["forward_price_F"])

k = 4
qubit_values = range(3, 9)

# Lognormal distribution parameters
mu = math.log(S0) + (r - q - 0.5 * sigma**2) * T
vol_T = sigma * math.sqrt(T)

def normal_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

def lognormal_cdf(s):
    if s <= 0:
        return 0.0
    z = (math.log(s) - mu) / vol_T
    return normal_cdf(z)

# Truncation bounds: plus/minus k standard deviations in log space
S_min = math.exp(mu - k * vol_T)
S_max = math.exp(mu + k * vol_T)

cdf_min = lognormal_cdf(S_min)
cdf_max = lognormal_cdf(S_max)
truncated_mass = cdf_max - cdf_min
tail_mass = 1.0 - truncated_mass

if truncated_mass <= 0:
    raise ValueError("Invalid truncation probability")

all_rows = []
summary = []

for n in qubit_values:
    N = 2**n

    # Uniform price grid in S
    S = np.linspace(S_min, S_max, N)

    # K is snapped to the nearest grid node
    K_index = int(np.argmin(np.abs(S - K)))
    K_grid = float(S[K_index])

    # Probability-cell boundaries
    boundaries = np.empty(N + 1)
    boundaries[0] = S_min
    boundaries[-1] = S_max
    boundaries[1:-1] = (S[:-1] + S[1:]) / 2

    # Probability in each cell under the truncated lognormal
    cdf_values = np.array([lognormal_cdf(x) for x in boundaries])
    raw_p = np.diff(cdf_values)
    p = raw_p / truncated_mass

    probability_sum = float(np.sum(p))

    if np.any(p < 0):
        raise ValueError(f"Negative probability found for n={n}")

    if not math.isclose(probability_sum, 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Probability sum failed for n={n}: {probability_sum}")

    if not np.any(np.isclose(S, K_grid, atol=1e-10)):
        raise ValueError(f"Strike is not a grid node for n={n}")

    for i in range(N):
        all_rows.append({
            "n": n,
            "index": i,
            "S": float(S[i]),
            "p": float(p[i]),
            "K_original": K,
            "K_grid": K_grid,
            "is_strike_node": bool(i == K_index)
        })

    summary.append({
        "n": n,
        "number_of_points": N,
        "S_min": float(S_min),
        "S_max": float(S_max),
        "grid_spacing": float(S[1] - S[0]),
        "K_original": K,
        "K_grid": K_grid,
        "K_grid_index": K_index,
        "K_snap_difference": K_grid - K,
        "probability_sum": probability_sum,
        "minimum_probability": float(np.min(p)),
        "maximum_probability": float(np.max(p)),
        "grid_mean": float(np.dot(S, p))
    })

# Save distribution CSV
distribution = pd.DataFrame(all_rows)
csv_path = DATA_FINAL / "person1_gbm_distribution_20261106.csv"
distribution.to_csv(csv_path, index=False)

# Save audit JSON
audit = {
    "as_of_date": inputs["as_of_date"],
    "expiry_date": inputs["expiry_date"],
    "measure": "risk_neutral",
    "distribution": "truncated_lognormal_from_risk_neutral_GBM",
    "S0": S0,
    "r": r,
    "q": q,
    "sigma": sigma,
    "T": T,
    "F": F,
    "k": k,
    "truncation_lower_bound": S_min,
    "truncation_upper_bound": S_max,
    "probability_mass_inside_range": truncated_mass,
    "probability_mass_outside_range": tail_mass,
    "probabilities_conditioned_on_truncation": True,
    "probability_cell_rule": "Voronoi cells bounded by adjacent grid midpoints; outer boundaries equal truncation limits",
    "grid_rule": "Uniformly spaced price points in S; K snapped to nearest node separately for each n",
    "qubit_configurations": summary,
    "validation": {
        "all_probability_sums_equal_one": True,
        "no_negative_probabilities": True,
        "strike_is_grid_node_for_all_n": True
    }
}

json_path = DATA_FINAL / "person1_distribution_audit_20261106.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=4)

# Save reproducible script
script_path = SCRIPT_DIR / "build_gbm_distribution.py"
print("DISTRIBUTION GENERATION COMPLETE")
print("CSV:", csv_path)
print("Audit:", json_path)
print("\nTruncation range:")
print(f"Lower: {S_min:.6f}")
print(f"Upper: {S_max:.6f}")
print(f"Probability inside: {truncated_mass:.10f}")
print(f"Probability outside: {tail_mass:.10f}")

print("\nGRID VALIDATION")
print(pd.DataFrame(summary).to_string(index=False))
