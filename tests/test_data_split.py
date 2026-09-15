import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_split import load_fixed_splits, validate_session_level_split


class DataSplitTests(unittest.TestCase):
    def test_fixed_assignments_are_preserved_and_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            split_paths = {}
            expected_hashes = {}
            expected_counts = {"development": 3, "validation": 2, "test": 2}
            first_id = 1000
            for name, count in expected_counts.items():
                path = Path(directory) / f"{name}.csv"
                frame = pd.DataFrame(
                    {
                        "Participant_ID": range(first_id, first_id + count),
                        "PHQ8_Score": range(count),
                    }
                )
                frame.to_csv(path, index=False)
                split_paths[name] = str(path)
                expected_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
                first_id += count

            splits = load_fixed_splits(
                split_paths,
                expected_counts=expected_counts,
                expected_hashes=expected_hashes,
            )
            self.assertEqual(
                splits["development"]["Participant_ID"].tolist(),
                [1000, 1001, 1002],
            )

    def test_changed_fixed_split_fingerprint_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for index, name in enumerate(["development", "validation", "test"]):
                path = Path(directory) / f"{name}.csv"
                pd.DataFrame(
                    {"Participant_ID": [index], "PHQ8_Score": [0]}
                ).to_csv(path, index=False)
                paths[name] = str(path)
            with self.assertRaises(ValueError):
                load_fixed_splits(
                    paths,
                    expected_counts={name: 1 for name in paths},
                    expected_hashes={name: "incorrect" for name in paths},
                )

    def test_overlap_fails_closed(self):
        development = pd.DataFrame({"Participant_ID": [1, 2], "PHQ8_Score": [0, 1]})
        test = pd.DataFrame({"Participant_ID": [2, 3], "PHQ8_Score": [1, 2]})
        with self.assertRaises(ValueError):
            validate_session_level_split({"development": development, "test": test})


if __name__ == "__main__":
    unittest.main()
