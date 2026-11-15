import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data" / "final" / "person1"


class TestPersonAOutputs(unittest.TestCase):

    def test_gbm_distribution_probabilities(self):
        df = pd.read_csv(FINAL / "person1_gbm_distribution_20261106.csv")

        self.assertIn("p", df.columns)
        self.assertIn("n", df.columns)
        self.assertTrue((df["p"] >= 0).all())

        for n, group in df.groupby("n"):
            self.assertAlmostEqual(group["p"].sum(), 1.0, places=10)

    def test_state_loader_probabilities(self):
        gbm = pd.read_csv(FINAL / "person1_gbm_distribution_20261106.csv")
        loader = pd.read_csv(
            FINAL / "person1_state_loader_probabilities_20261106.csv"
        )

        for n in sorted(set(gbm["n"]) & set(loader["n"])):
            expected = gbm.loc[gbm["n"] == n, "p"].to_numpy()
            target = loader.loc[loader["n"] == n, "p_target"].to_numpy()
            actual = loader.loc[loader["n"] == n, "p_actual"].to_numpy()

            self.assertEqual(len(expected), len(target))
            self.assertEqual(len(target), len(actual))

            np.testing.assert_allclose(target, expected, rtol=0, atol=1e-12)
            np.testing.assert_allclose(actual, target, rtol=0, atol=2e-10)

    def test_valuation_output(self):
        df = pd.read_csv(
            FINAL / "person1_option_valuation_20261106.csv"
        )

        required = {
            "n", "qubits", "number_of_states",
            "actual_strike", "snapped_strike",
            "V_mkt_close", "V_mkt_settlement",
            "V_BS_actual_strike", "V_BS_snapped_strike",
            "V_grid", "V_load",
            "grid_minus_BS_snapped", "load_minus_grid",
            "max_probability_error"
        }

        self.assertTrue(required.issubset(df.columns))
        self.assertEqual(len(df), 6)
        self.assertNotIn("grid_minus_market", df.columns)
        self.assertNotIn("load_minus_market", df.columns)

        np.testing.assert_allclose(
            df["V_load"], df["V_grid"], rtol=0, atol=1e-8
        )

    def test_gbm_input_reference(self):
        data = json.loads(
            (FINAL / "person1_gbm_inputs_20261106.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(data["as_of_date"], "2026-11-06")
        self.assertEqual(data["expiry_date"], "2026-12-08")
        self.assertEqual(data["calendar_days_to_expiry"], 32)
        self.assertAlmostEqual(data["spot_price_S0"], 25030.35, places=2)
        self.assertAlmostEqual(
            data["primary_sigma"],
            data["volatility_annualized"]["90"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
