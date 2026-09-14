import unittest

import numpy as np
import pandas as pd

from experiments.evaluate import (
    paired_t_test,
    pairwise_comparisons,
    participant_error_table,
    summarize_results,
)


def fixture_results():
    rows = []
    for configuration, predictions in {
        "direct": {1: [7, 9, 8], 2: [12, 10, 11]},
        "multi": {1: [6, 6, 6], 2: [9, 9, 9]},
    }.items():
        for participant_id, values in predictions.items():
            true_score = 5 if participant_id == 1 else 10
            for seed, prediction in zip([11, 22, 33], values):
                rows.append(
                    {
                        "configuration": configuration,
                        "participant_id": participant_id,
                        "seed": seed,
                        "true_score": true_score,
                        "predicted_score": prediction,
                        "audit_decision": "PASS" if configuration == "multi" else "",
                        "audit_corrected": False,
                        "audit_unsupported_inference": False,
                        "knowledge_reference_ids": "phq8_interest" if configuration == "multi" else "",
                        "rationale": "input evidence" if configuration == "multi" else "",
                    }
                )
    return pd.DataFrame(rows)


class EvaluationTests(unittest.TestCase):
    def test_participant_is_independent_unit(self):
        direct = fixture_results().query("configuration == 'direct'")
        participant = participant_error_table(direct)
        self.assertEqual(len(participant), 2)
        self.assertTrue((participant["completed_seeds"] == 3).all())

    def test_summary_reports_participants_and_runs_separately(self):
        summary = summarize_results(fixture_results(), bootstrap_iterations=50)
        self.assertTrue((summary["participant_count"] == 2).all())
        self.assertTrue((summary["completed_runs"] == 6).all())

    def test_pairwise_test_uses_two_participants_not_six_seed_rows(self):
        comparisons = pairwise_comparisons(fixture_results())
        self.assertEqual(int(comparisons.iloc[0]["paired_participants"]), 2)

    def test_constant_nonzero_paired_difference_is_not_reported_as_null(self):
        result = paired_t_test(np.array([2.0, 2.0]), np.array([1.0, 1.0]))
        self.assertEqual(result["p_value"], 0.0)
        self.assertTrue(np.isinf(result["t_statistic"]))


if __name__ == "__main__":
    unittest.main()
