"""Fast structural checks; run after train.py and score.py."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from train import features
from score import validate_december, validate_predictions

ROOT = Path(__file__).resolve().parent


class SubmissionTests(unittest.TestCase):
    def test_submission_contract(self):
        p = pd.read_csv(ROOT / "validation_predictions.csv")
        validate_predictions(p)
        template = pd.read_csv(ROOT / "data/validation_predictions_template.csv")
        self.assertEqual(p.load_id.tolist(), template.load_id.tolist())

    def test_december_contract(self):
        d = pd.read_csv(ROOT / "data/december_chart_inputs.csv")
        validate_december(d)
        self.assertTrue((ROOT / "scorer_results/candidate_december.png").is_file())

    def test_chart_needs_no_hidden_features(self):
        d = pd.read_csv(ROOT / "data/december_chart_inputs.csv")
        x = features(d, "chart")
        self.assertFalse({"load_id", "posted_rate", "predicted_rate",
                          "quote_signal", "market_index"} & set(x.columns))
        self.assertTrue(np.isfinite(x.select_dtypes("number")).all().all())

    def test_invalid_weight_is_flagged_not_made_positive(self):
        d = pd.read_csv(ROOT / "data/december_chart_inputs.csv").head(3).copy()
        d["weight"] = [-100, np.nan, 32000]
        x = features(d, "chart")
        self.assertEqual(x.weight_invalid.tolist(), [1, 1, 0])
        self.assertEqual(x.weight.tolist(), [-999.0, -999.0, 32000.0])

    def test_core_excludes_quote_and_ids(self):
        d = pd.read_csv(ROOT / "data/validation.csv", nrows=3)
        x = features(d, "core")
        self.assertNotIn("quote_signal", x)
        self.assertNotIn("load_id", x)


if __name__ == "__main__":
    unittest.main()