import unittest

from src.evaluation.metrics import exact_match_accuracy


class TestExactMatchAccuracy(unittest.TestCase):

    def test_all_correct(self):
        references = ["I need water", "Thank you"]
        predictions = ["I need water", "Thank you"]

        self.assertEqual(exact_match_accuracy(references, predictions), 1.0)

    def test_partial_correct(self):
        references = ["I need water", "Thank you"]
        predictions = ["I want water", "Thank you"]

        self.assertEqual(exact_match_accuracy(references, predictions), 0.5)

    def test_case_insensitive(self):
        references = ["Thank You"]
        predictions = ["thank you"]

        self.assertEqual(exact_match_accuracy(references, predictions), 1.0)

    def test_empty_input(self):
        self.assertEqual(exact_match_accuracy([], []), 0.0)

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            exact_match_accuracy(["Hello"], [])
