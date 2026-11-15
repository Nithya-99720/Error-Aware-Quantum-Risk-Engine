import json
import math
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import qpy
from qiskit.quantum_info import Statevector


ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "data" / "final" / "person1"
PROCESSED = ROOT / "data" / "processed"

DIST_FILE = FINAL / "person1_gbm_distribution_20261106.csv"
INPUT_FILE = FINAL / "person1_gbm_inputs_20261106.json"
OPTIONS_FILE = PROCESSED / "options_clean_20261106.csv"
QPY_FILE = FINAL / "person1_state_preparation_20261106.qpy"

CSV_OUT = FINAL / "person1_option_valuation_20261106.csv"
JSON_OUT = FINAL / "person1_option_valuation_audit_20261106.json"

PRIMARY_STRIKE = 25150.0
OPTION_TYPE = "CE"
TOLERANCE = 1e-9


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call(S0, K, r, q, sigma, T):
    d1 = (
        math.log(S0 / K) +
        (r - q + 0.5 * sigma**2) * T
    ) / (sigma * math.sqrt(T))

    d2 = d1 - sigma * math.sqrt(T)

    return (
        S0 * math.exp(-q * T) * norm_cdf(d1)
        - K * math.exp(-r * T) * norm_cdf(d2)
    )


def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        inputs = json.load(f)

    distribution = pd.read_csv(DIST_FILE)
    options = pd.read_csv(OPTIONS_FILE)

    S0 = float(inputs["spot_price_S0"])
    r = float(inputs["continuous_risk_free_rate_r"])
    q = float(inputs["dividend_yield_decimal_q"])
    sigma = float(inputs["primary_sigma"])
    T = float(inputs["time_to_expiry_years"])

    market_rows = options[
        (options["strike"].astype(float) == PRIMARY_STRIKE) &
        (options["option_type"] == OPTION_TYPE) &
        (options["expiry"].astype(str) == inputs["expiry_date"])
    ]

    if len(market_rows) != 1:
        raise ValueError(
            f"Expected one primary option contract; found {len(market_rows)}"
        )

    market = market_rows.iloc[0]

    V_mkt_close = float(market["close_price"])
    V_mkt_settlement = float(market["settlement_price"])

    V_BS_actual = black_scholes_call(
        S0, PRIMARY_STRIKE, r, q, sigma, T
    )

    with open(QPY_FILE, "rb") as f:
        circuits = qpy.load(f)

    circuits_by_n = {}

    for circuit in circuits:
        n = circuit.num_qubits

        if n in circuits_by_n:
            raise ValueError(f"Duplicate QPY circuit for n={n}")

        circuits_by_n[n] = circuit

    results = []

    for n in sorted(distribution["n"].unique()):
        n = int(n)

        subset = (
            distribution[distribution["n"] == n]
            .sort_values("index")
            .reset_index(drop=True)
        )

        if n not in circuits_by_n:
            raise ValueError(f"Missing saved circuit for n={n}")

        S = subset["S"].to_numpy(dtype=float)
        p_target = subset["p"].to_numpy(dtype=float)

        K_values = subset["K_grid"].unique()

        if len(K_values) != 1:
            raise ValueError(f"Inconsistent snapped strike for n={n}")

        K_grid = float(K_values[0])

        circuit = circuits_by_n[n]
        state = Statevector.from_instruction(circuit)
        p_actual = np.abs(state.data) ** 2

        if len(p_actual) != len(S):
            raise ValueError(f"State dimension mismatch for n={n}")

        probability_error = float(
            np.max(np.abs(p_actual - p_target))
        )

        if probability_error > TOLERANCE:
            raise AssertionError(
                f"Loader probability error too large for n={n}"
            )

        # Grid price uses the target GBM probabilities.
        payoff = np.maximum(S - K_grid, 0.0)

        V_grid = math.exp(-r * T) * float(
            np.dot(p_target, payoff)
        )

        # Loaded price uses probabilities reconstructed from QPY.
        V_load = math.exp(-r * T) * float(
            np.dot(p_actual, payoff)
        )

        # Same-strike BS benchmark isolates grid discretization.
        V_BS_grid_strike = black_scholes_call(
            S0, K_grid, r, q, sigma, T
        )

        results.append({
            "n": n,
            "qubits": n,
            "number_of_states": len(S),
            "actual_strike": PRIMARY_STRIKE,
            "snapped_strike": K_grid,
            "strike_snap_difference": K_grid - PRIMARY_STRIKE,
            "V_mkt_close": V_mkt_close,
            "V_mkt_settlement": V_mkt_settlement,
            "V_BS_actual_strike": V_BS_actual,
            "V_BS_snapped_strike": V_BS_grid_strike,
            "V_grid": V_grid,
            "V_load": V_load,
            "grid_minus_BS_snapped": V_grid - V_BS_grid_strike,
            "load_minus_grid": V_load - V_grid,
            "max_probability_error": probability_error
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(CSV_OUT, index=False, float_format="%.15g")

    audit = {
        "as_of_date": inputs["as_of_date"],
        "expiry_date": inputs["expiry_date"],
        "option_type": OPTION_TYPE,
        "primary_strike": PRIMARY_STRIKE,
        "market_price_convention": "NSE bhavcopy close price",
        "market_close_price": V_mkt_close,
        "market_settlement_price": V_mkt_settlement,
        "S0": S0,
        "r": r,
        "q": q,
        "sigma": sigma,
        "T": T,
        "V_BS_actual_strike": V_BS_actual,
        "distribution_source": str(DIST_FILE),
        "QPY_source": str(QPY_FILE),
        "probability_tolerance": TOLERANCE,
        "pricing_method": (
            "Discounted expected European call payoff "
            "on the truncated, conditionally normalized GBM grid"
        ),
        "V_grid_method": "Target distribution probabilities",
        "V_load_method": (
            "Probabilities reconstructed from saved QPY circuits "
            "using exact Statevector simulation"
        ),
        "configurations": results,
        "all_loader_checks_passed": all(
            x["max_probability_error"] <= TOLERANCE
            for x in results
        ),
        "input_sha256": {
            "distribution": sha256(DIST_FILE),
            "gbm_inputs": sha256(INPUT_FILE),
            "options": sha256(OPTIONS_FILE),
            "qpy": sha256(QPY_FILE)
        },
        "output_csv": str(CSV_OUT)
    }

    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=4)

    print("\nPERSON A OPTION VALUATION COMPLETE")
    print(f"Market close: ₹{V_mkt_close:.6f}")
    print(f"Market settlement: ₹{V_mkt_settlement:.6f}")
    print(f"Black-Scholes at actual strike: ₹{V_BS_actual:.6f}")
    print("\nValuation comparison:")
    print(result_df.to_string(index=False))
    print(f"\nCSV: {CSV_OUT}")
    print(f"Audit: {JSON_OUT}")


if __name__ == "__main__":
    main()
