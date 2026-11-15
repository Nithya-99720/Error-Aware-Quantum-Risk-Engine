import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile, qpy
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector


ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "data" / "final" / "person1"

SOURCE = FINAL / "person1_gbm_distribution_20261106.csv"
QPY_OUTPUT = FINAL / "person1_state_preparation_20261106.qpy"
CSV_OUTPUT = FINAL / "person1_state_loader_probabilities_20261106.csv"
JSON_OUTPUT = FINAL / "person1_state_loader_audit_20261106.json"

TOLERANCE = 1e-9
SEED = 42


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    FINAL.mkdir(parents=True, exist_ok=True)

    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    df = pd.read_csv(SOURCE)

    required = {"n", "index", "S", "p", "K_original", "K_grid",
                "is_strike_node"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    circuits = []
    probability_rows = []
    configuration_reports = []

    for n in sorted(df["n"].unique()):
        n = int(n)
        subset = df[df["n"] == n].sort_values("index").reset_index(drop=True)

        if n < 1 or len(subset) != 2 ** n:
            raise ValueError(f"Invalid grid size for n={n}")

        indices = subset["index"].to_numpy(dtype=int)

        if not np.array_equal(indices, np.arange(2 ** n)):
            raise ValueError(f"Invalid state index sequence for n={n}")

        p = subset["p"].to_numpy(dtype=float)
        strikes = subset["is_strike_node"].astype(bool).to_numpy()

        if not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError(f"Invalid probabilities for n={n}")

        if not np.isclose(p.sum(), 1.0, atol=TOLERANCE):
            raise ValueError(f"Probability sum is not one for n={n}")

        if np.count_nonzero(strikes) != 1:
            raise ValueError(f"Expected exactly one strike node for n={n}")

        # Exact state-preparation target amplitudes.
        amplitudes = np.sqrt(p)

        qc = QuantumCircuit(n, name=f"GBM_StatePrep_n{n}")
        qc.append(StatePreparation(amplitudes), range(n))

        qc.metadata = {
            "n": n,
            "number_of_points": 2 ** n,
            "source": SOURCE.name,
            "measure": "risk_neutral",
            "loader": "Qiskit StatePreparation"
        }

        # Exact statevector verification; no shot noise.
        state = Statevector.from_instruction(qc)
        p_actual = np.abs(state.data) ** 2

        errors = np.abs(p_actual - p)
        max_error = float(errors.max())

        if max_error > TOLERANCE:
            raise AssertionError(
                f"State preparation failed for n={n}: "
                f"maximum error={max_error}"
            )

        # Resource accounting in the u/cx basis.
        compiled = transpile(
            qc,
            basis_gates=["u", "cx"],
            optimization_level=0,
            seed_transpiler=SEED
        )

        ops = {str(k): int(v) for k, v in compiled.count_ops().items()}

        circuits.append(qc)

        for i in range(len(p)):
            probability_rows.append({
                "n": n,
                "index": int(i),
                "S": float(subset.loc[i, "S"]),
                "p_target": float(p[i]),
                "p_actual": float(p_actual[i]),
                "absolute_error": float(errors[i])
            })

        configuration_reports.append({
            "n": n,
            "number_of_points": int(2 ** n),
            "probability_sum": float(p.sum()),
            "probability_verification": "PASS",
            "max_absolute_probability_error": max_error,
            "statevector_verification": "PASS",
            "strike_grid_index": int(np.flatnonzero(strikes)[0]),
            "transpilation_basis": ["u", "cx"],
            "optimization_level": 0,
            "gate_counts": ops,
            "total_gates": int(compiled.size()),
            "two_qubit_gates": int(compiled.count_ops().get("cx", 0)),
            "circuit_depth": int(compiled.depth())
        })

        print(
            f"n={n}: points={2**n}, "
            f"max probability error={max_error:.3e}, "
            f"depth={compiled.depth()}, "
            f"2-qubit gates={compiled.count_ops().get('cx', 0)}"
        )

    if len(circuits) != 6:
        raise ValueError(
            f"Expected six configurations (n=3..8); found {len(circuits)}"
        )

    # Save all six circuits together in QPY format.
    with open(QPY_OUTPUT, "wb") as f:
        qpy.dump(circuits, f)

    # Save target versus actual probabilities.
    probability_df = pd.DataFrame(probability_rows)
    probability_df.to_csv(CSV_OUTPUT, index=False, float_format="%.17g")

    report = {
        "source_file": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "loader": "Qiskit StatePreparation",
        "state_definition": "amplitude_i = sqrt(p_i)",
        "verification_method": "Exact Qiskit Statevector",
        "tolerance": TOLERANCE,
        "transpilation_basis": ["u", "cx"],
        "optimization_level": 0,
        "transpiler_seed": SEED,
        "configurations": configuration_reports,
        "all_configurations_passed": all(
            x["statevector_verification"] == "PASS"
            for x in configuration_reports
        ),
        "outputs": {
            "qpy_file": str(QPY_OUTPUT),
            "probability_csv": str(CSV_OUTPUT)
        }
    }

    report["output_sha256"] = {
        "qpy": sha256(QPY_OUTPUT),
        "probability_csv": sha256(CSV_OUTPUT)
    }

    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\nSTATE LOADER BUILD COMPLETE")
    print(f"Circuits saved: {QPY_OUTPUT}")
    print(f"Probabilities saved: {CSV_OUTPUT}")
    print(f"Audit saved: {JSON_OUTPUT}")
    print("All six configurations passed exact statevector verification.")


if __name__ == "__main__":
    main()
