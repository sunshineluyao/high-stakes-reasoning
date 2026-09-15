import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments.runner import normalize_prediction, run_audit_agent


class RunnerTests(unittest.TestCase):
    def test_null_rationale_is_normalized_to_empty_text(self):
        prediction = normalize_prediction({"individual_scores": {}, "rationale": None})
        self.assertEqual(prediction["rationale"], "")

    def test_rejected_audit_requires_a_corrected_prediction(self):
        class FakeLLM:
            def invoke(self, messages):
                return SimpleNamespace(
                    content=(
                        '{"decision":"REJECT","corrected_prediction":null,'
                        '"reason":"unsupported","unsupported_inference":true}'
                    )
                )

        with patch("experiments.runner.system_message", side_effect=lambda content: content):
            with self.assertRaises(ValueError):
                run_audit_agent({}, "observation", "knowledge", FakeLLM())


if __name__ == "__main__":
    unittest.main()
