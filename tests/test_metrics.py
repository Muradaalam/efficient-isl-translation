import unittest
from src.evaluation.metrics import exact_match_accuracy, word_error_rate, bleu_scores

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


class TestWordErrorRate(unittest.TestCase):

    def test_perfect_prediction(self):
        self.assertEqual(word_error_rate(["I need water"], ["I need water"]), 0.0)

    def test_substitution(self):
        self.assertAlmostEqual(word_error_rate(["I need water"], ["I want water"]), 1 / 3)

    def test_deletion(self):
        self.assertAlmostEqual(word_error_rate(["I need water"], ["I water"]), 1 / 3)

    def test_insertion(self):
        self.assertAlmostEqual(word_error_rate(["I need water"], ["I really need water"]), 1 / 3)

    def test_multiple_errors(self):
        self.assertAlmostEqual(word_error_rate(["I need water"], ["You want water"]), 2 / 3)

    def test_empty_sentences(self):
        self.assertEqual(word_error_rate([""], [""]), 0.0)

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            word_error_rate(["Hello"], [])


class TestBLEUScore(unittest.TestCase):

    def test_perfect_prediction(self):
        scores = bleu_scores(["I am going home"], ["I am going home"])

        for n in range(1, 5):
            self.assertAlmostEqual(scores[f"bleu_{n}"], 1.0)

    def test_partial_prediction(self):
        scores = bleu_scores(["I am going home"], ["I am going school"])

        self.assertAlmostEqual(scores["bleu_1"], 0.75)
        self.assertAlmostEqual(scores["bleu_2"], 0.70710678)
        self.assertAlmostEqual(scores["bleu_3"], 0.62996052)
        self.assertEqual(scores["bleu_4"], 0.0)

    def test_empty_predictions(self):
        scores = bleu_scores(["I am happy"], [""])

        for value in scores.values():
            self.assertEqual(value, 0.0)

    def test_case_insensitive(self):
        scores = bleu_scores(["I AM HAPPY TODAY"], ["i am happy today"])

        self.assertAlmostEqual(scores["bleu_4"], 1.0)

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            bleu_scores(["Hello"], [])

    def test_short_prediction(self):
        scores = bleu_scores(["I am going home"], ["I am"])

        self.assertLess(scores["bleu_1"], 1.0)
        self.assertEqual(scores["bleu_4"], 0.0)


