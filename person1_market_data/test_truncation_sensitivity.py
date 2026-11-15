
from pathlib import Path
import json
import math
import hashlib

import numpy as np
import pandas as pd

# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "data" / "final" / "person1"
OUTPUT = ROOT / "data" / "experiments" / "person1_truncation"

OUTPUT.mkdir(parents=True, exist_ok=True)

INPUT_FILE = FINAL / "person1_gbm_inputs_20261106.json"
STRIKE_FILE = FINAL / "person1_selected_strikes_20261106.csv"

K_VALUES = [3, 4, 5]
QUBIT_VALUES = [3, 4, 5, 6, 7, 8]

# --------------------------------------------------
# LOAD INPUTS
# --------------------------------------------------

with open(INPUT_FILE, encoding="utf-8") as f:
    inputs = json.load(f)

strikes = pd.read_csv(STRIKE_FILE)
atm = strikes[strikes["selection"] == "ATM"]

atm_strikes = atm["selected_strike"].astype(float).unique()

if len(atm_strikes) != 1:
    raise ValueError(
        f"Expected exactly one unique ATM strike, found {atm_strikes}"
    )

K = float(atm_strikes[0])

S0 = float(inputs["spot_price_S0"])
r = float(inputs["continuous_risk_free_rate_r"])
q = float(inputs["dividend_yield_decimal_q"])
sigma = float(inputs["primary_sigma"])
T = float(inputs["time_to_expiry_years"])

mu = math.log(S0) + (r - q - 0.5 * sigma**2) * T
vol_T = sigma * math.sqrt(T)

# --------------------------------------------------
# MATHEMATICAL FUNCTIONS
# --------------------------------------------------

def normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def lognormal_cdf(s):
    if s <= 0:
        return 0.0
    return normal_cdf((math.log(s) - mu) / vol_T)


def black_scholes_call(strike):
    d1 = (
        math.log(S0 / strike)
        + (r - q + 0.5 * sigma**2) * T
    ) / vol_T

    d2 = d1 - vol_T

    return (
        S0 * math.exp(-q * T) * normal_cdf(d1)
        - strike * math.exp(-r * T) * normal_cdf(d2)
    )


def continuous_truncated_call(strike, lower, upper, mass):

    if not lower < strike < upper:
        raise ValueError("Strike is outside truncation interval")

    d1_strike = (
        math.log(S0 / strike)
        + (r - q + 0.5 * sigma**2) * T
    ) / vol_T

    d1_upper = (
        math.log(S0 / upper)
        + (r - q + 0.5 * sigma**2) * T
    ) / vol_T

    d2_strike = (
        math.log(S0 / strike)
        + (r - q - 0.5 * sigma**2) * T
    ) / vol_T

    d2_upper = (
        math.log(S0 / upper)
        + (r - q - 0.5 * sigma**2) * T
    ) / vol_T

    expected_stock_component = (
        S0 * math.exp((r - q) * T)
        * (normal_cdf(d1_strike) - normal_cdf(d1_upper))
    )

    expected_strike_component = (
        strike
        * (normal_cdf(d2_strike) - normal_cdf(d2_upper))
    )

    return (
        math.exp(-r * T)
        * (expected_stock_component - expected_strike_component)
        / mass
    )


# --------------------------------------------------
# PROTECT EXISTING VALIDATED FILES
# --------------------------------------------------

protected_files = [
    FINAL / "person1_gbm_distribution_20261106.csv",
    FINAL / "person1_distribution_audit_20261106.json",
    FINAL / "person1_state_preparation_20261106.qpy",
]

baseline_hashes = {}

for path in protected_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Protected baseline file missing: {path}"
        )

    baseline_hashes[str(path.relative_to(ROOT))] = (
        hashlib.sha256(path.read_bytes()).hexdigest()
    )


# --------------------------------------------------
# RUN ALL 18 CONFIGURATIONS
# --------------------------------------------------

results = []

for n_qubits in QUBIT_VALUES:

    n_points = 2 ** n_qubits

    for k in K_VALUES:

        lower = math.exp(mu - k * vol_T)
        upper = math.exp(mu + k * vol_T)

        mass = lognormal_cdf(upper) - lognormal_cdf(lower)
        tail_mass = 1.0 - mass

        if not 0 < mass <= 1:
            raise ValueError(
                f"Invalid retained probability: k={k}, n={n_qubits}"
            )

        # Uniform price grid
        S = np.linspace(lower, upper, n_points)

        # Snap strike to nearest grid point
        strike_index = int(np.argmin(np.abs(S - K)))
        K_grid = float(S[strike_index])

        # Voronoi cell boundaries
        boundaries = np.empty(n_points + 1)
        boundaries[0] = lower
        boundaries[-1] = upper
        boundaries[1:-1] = (S[:-1] + S[1:]) / 2

        cdf_values = np.array([
            lognormal_cdf(x) for x in boundaries
        ])

        raw_p = np.diff(cdf_values)
        p = raw_p / mass

        probability_sum = float(np.sum(p))

        if np.any(p < 0):
            raise ValueError(
                f"Negative probability: k={k}, n={n_qubits}"
            )

        if not math.isclose(
            probability_sum, 1.0, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError(
                f"Probability sum failed: k={k}, n={n_qubits}"
            )

        # Grid valuation
        discount = math.exp(-r * T)
        payoff = np.maximum(S - K_grid, 0)

        V_grid = float(discount * np.dot(p, payoff))

        # Continuous conditional benchmark
        V_cont = continuous_truncated_call(
            K_grid, lower, upper, mass
        )

        # Full Black-Scholes benchmark at same snapped strike
        V_BS = black_scholes_call(K_grid)

        # Error decomposition
        discretization_floor = abs(V_grid - V_cont)
        truncation_difference = abs(V_BS - V_cont)
        total_difference = abs(V_BS - V_grid)

        if not math.isclose(
            total_difference,
            discretization_floor + truncation_difference,
            rel_tol=1e-7,
            abs_tol=1e-8
        ):
            raise ValueError(
                f"Error decomposition failed: k={k}, n={n_qubits}"
            )

        # Save individual distribution
        distribution = pd.DataFrame({
            "index": np.arange(n_points),
            "S": S,
            "p": p,
            "K_original": K,
            "K_grid": K_grid,
            "is_strike_node": np.arange(n_points) == strike_index
        })

        distribution_path = (
            OUTPUT / f"distribution_k{k}_n{n_qubits}.csv"
        )

        distribution.to_csv(distribution_path, index=False)

        # Save summary row
        results.append({
            "k": k,
            "qubits": n_qubits,
            "number_of_points": n_points,
            "lower_bound": lower,
            "upper_bound": upper,
            "probability_inside": mass,
            "probability_outside": tail_mass,
            "grid_spacing": float(S[1] - S[0]),
            "actual_strike": K,
            "snapped_strike": K_grid,
            "strike_snap_difference": K_grid - K,
            "probability_sum": probability_sum,
            "V_grid": V_grid,
            "V_continuous_truncated": V_cont,
            "V_BS_full": V_BS,
            "discretization_floor": discretization_floor,
            "truncation_conditioning_difference": truncation_difference,
            "total_difference_from_full_BS": total_difference,
            "distribution_file": distribution_path.name
        })

        print(
            f"Completed: n={n_qubits}, k={k}, "
            f"total error=Rs.{total_difference:.8f}"
        )


# --------------------------------------------------
# SAVE SUMMARY AND AUDIT
# --------------------------------------------------

result_df = pd.DataFrame(results)

if len(result_df) != 18:
    raise RuntimeError(
        f"Expected 18 configurations, got {len(result_df)}"
    )

CSV_OUT = (
    OUTPUT / "truncation_sensitivity_k3_k4_k5_n3_to_n8.csv"
)

JSON_OUT = (
    OUTPUT / "truncation_sensitivity_k3_k4_k5_n3_to_n8_audit.json"
)

result_df.to_csv(CSV_OUT, index=False)

audit = {
    "experiment": "GBM truncation sensitivity",
    "as_of_date": inputs["as_of_date"],
    "expiry_date": inputs["expiry_date"],
    "S0": S0,
    "r": r,
    "q": q,
    "sigma": sigma,
    "T": T,
    "original_strike": K,
    "qubit_values": QUBIT_VALUES,
    "truncation_values": K_VALUES,
    "number_of_configurations": len(results),
    "probability_method": "Conditional normalization within truncation interval",
    "pricing_method": "Discounted European call payoff",
    "baseline_files_sha256_before": baseline_hashes,
    "results": results,
    "validation": {
        "all_probability_sums_equal_one": True,
        "no_negative_probabilities": True,
        "strike_is_grid_node": True,
        "error_decomposition_consistent": True
    }
}

with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=4)


# --------------------------------------------------
# VERIFY PROTECTED FILES
# --------------------------------------------------

for relative_path, expected_hash in baseline_hashes.items():

    path = ROOT / relative_path
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Protected baseline file changed: {relative_path}"
        )


# --------------------------------------------------
# FINAL REPORT
# --------------------------------------------------

print("\nTRUNCATION SENSITIVITY EXPERIMENT COMPLETE")
print(f"Configurations tested: {len(results)}")
print(result_df.to_string(index=False))
print("\nCSV:", CSV_OUT)
print("AUDIT:", JSON_OUT)
print("\nPASS: All 18 configurations completed.")
print("PASS: Existing validated baseline files remain unchanged.")
