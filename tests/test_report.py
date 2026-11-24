
import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.report import save_evaluation_report


class TestEvaluationReport(unittest.TestCase):

    def setUp(self):
        self.sample_ids = ["sample_001", "sample_002"]
        self.references = ["I need water", "I am happy"]
        self.predictions = ["I need water", "I happy"]

    def test_report_files_created(self):
        with tempfile.TemporaryDirectory() as directory:
            report = save_evaluation_report(
                self.sample_ids, self.references, self.predictions,
                directory, "uniform_mock"
            )

            self.assertTrue((Path(directory) / "predictions.csv").exists())
            self.assertTrue((Path(directory) / "metrics.json").exists())
            self.assertEqual(report["num_samples"], 2)

    def test_csv_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            save_evaluation_report(
                self.sample_ids, self.references, self.predictions,
                directory, "uniform_mock"
            )

            with open(Path(directory) / "predictions.csv", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["sample_id"], "sample_001")
            self.assertEqual(rows[1]["prediction"], "I happy")

    def test_json_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            save_evaluation_report(
                self.sample_ids, self.references, self.predictions,
                directory, "uniform_mock"
            )

            with open(Path(directory) / "metrics.json") as file:
                data = json.load(file)

            self.assertEqual(data["experiment_id"], "uniform_mock")
            self.assertEqual(data["split"], "test")
            self.assertEqual(data["exact_match"], 0.5)

    def test_mismatched_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_evaluation_report(
                    ["sample_001"], self.references, self.predictions,
                    directory, "uniform_mock"
                )

    def test_duplicate_sample_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_evaluation_report(
                    ["sample_001", "sample_001"],
                    self.references, self.predictions,
                    directory, "uniform_mock"
                )

    def test_empty_report(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_evaluation_report([], [], [], directory, "uniform_mock")
