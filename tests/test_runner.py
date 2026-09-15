import unittest

from experiments.runner import normalize_prediction


class RunnerTests(unittest.TestCase):
    def test_null_rationale_is_normalized_to_empty_text(self):
        prediction = normalize_prediction({"individual_scores": {}, "rationale": None})
        self.assertEqual(prediction["rationale"], "")


if __name__ == "__main__":
    unittest.main()
