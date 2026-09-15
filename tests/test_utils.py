import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.utils import _summarize_action_units


class UtilityTests(unittest.TestCase):
    def test_missing_action_unit_column_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "action_units.csv"
            pd.DataFrame({"success": [1], "AU12_r": [0.2]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                _summarize_action_units(path)


if __name__ == "__main__":
    unittest.main()
