
import unittest

from src.evaluation.evaluate import evaluate_predictions


class TestEvaluationPipeline(unittest.TestCase):

    def test_perfect_predictions(self):
        references = ["I need water", "I am happy today"]
        predictions = ["I need water", "I am happy today"]

        results = evaluate_predictions(references, predictions)

        self.assertEqual(results["num_samples"], 2)
        self.assertEqual(results["exact_match"], 1.0)
        self.assertEqual(results["wer"], 0.0)
        self.assertAlmostEqual(results["bleu_4"], 1.0)

    def test_partial_predictions(self):
        references = ["I need water", "I am happy"]
        predictions = ["I want water", "I am happy"]

        results = evaluate_predictions(references, predictions)

        self.assertEqual(results["num_samples"], 2)
        self.assertEqual(results["exact_match"], 0.5)
        self.assertAlmostEqual(results["wer"], 1 / 6)

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            evaluate_predictions(["Hello"], [])

    def test_empty_dataset(self):
        with self.assertRaises(ValueError):
            evaluate_predictions([], [])
