import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "featured_data" / "master_with_target.csv"
NOTEBOOK = ROOT / "notebooks" / "models" / "GNN_final_single_path_executable_ready_executed (1).ipynb"

class GNNSetupTests(unittest.TestCase):
    def test_repository_dataset_and_required_columns_exist(self):
        self.assertTrue(DATA.is_file(), f"Missing prepared data file: {DATA}")
        with DATA.open(newline="", encoding="utf-8-sig") as stream:
            columns = next(csv.reader(stream))
        self.assertIn("Date", columns)
        expected = [f"{name}_Target" for name in [
            "NIFTY_AUTO", "NIFTY_BANK", "NIFTY_ENERGY", "NIFTY_FMCG",
            "NIFTY_IT", "NIFTY_METAL", "NIFTY_PHARMA", "NIFTY_REALTY"
        ]]
        self.assertEqual([], [name for name in expected if name not in columns])

    def test_notebook_has_repository_relative_path_fallback(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"] if cell.get("cell_type") == "code"
        )
        self.assertIn('root / "Data" / "featured_data" / "master_with_target.csv"', code)
        self.assertIn("GNN_DATA_PATH", code)
        self.assertIn("Using prepared dataset:", code)

if __name__ == "__main__":
    unittest.main()
