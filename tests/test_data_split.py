import unittest

import pandas as pd

from src.data_split import split_sessions, validate_session_level_split


class DataSplitTests(unittest.TestCase):
    def test_expected_counts_and_no_overlap(self):
        sessions = pd.DataFrame(
            {"Participant_ID": range(1000, 1184), "PHQ8_Score": [index % 25 for index in range(184)]}
        )
        splits = split_sessions(sessions, seed=42)
        self.assertEqual({name: len(split) for name, split in splits.items()}, {
            "development": 128,
            "validation": 19,
            "test": 37,
        })
        validate_session_level_split(splits)
        all_ids = [
            participant_id
            for split in splits.values()
            for participant_id in split["Participant_ID"].tolist()
        ]
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_overlap_fails_closed(self):
        development = pd.DataFrame({"Participant_ID": [1, 2], "PHQ8_Score": [0, 1]})
        test = pd.DataFrame({"Participant_ID": [2, 3], "PHQ8_Score": [1, 2]})
        with self.assertRaises(ValueError):
            validate_session_level_split({"development": development, "test": test})


if __name__ == "__main__":
    unittest.main()
