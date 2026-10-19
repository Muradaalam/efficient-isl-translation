import unittest
import torch

from src.inference.predict import decode_tokens, predict_tokens


class TestDecodeTokens(unittest.TestCase):

    def setUp(self):
        self.vocab = {
            0: "<pad>",
            1: "<sos>",
            2: "<eos>",
            3: "i",
            4: "need",
            5: "water"
        }
        self.special_ids = {0, 1, 2}

    def test_normal_decoding(self):
        result = decode_tokens([1, 3, 4, 5, 2], self.vocab, 2, self.special_ids)
        self.assertEqual(result, "i need water")

    def test_stop_at_eos(self):
        result = decode_tokens([1, 3, 2, 4, 5], self.vocab, 2, self.special_ids)
        self.assertEqual(result, "i")

    def test_skip_padding(self):
        result = decode_tokens([0, 1, 3, 4, 5, 2, 0], self.vocab, 2, self.special_ids)
        self.assertEqual(result, "i need water")

    def test_unknown_token(self):
        with self.assertRaises(ValueError):
            decode_tokens([1, 99, 2], self.vocab, 2, self.special_ids)

    def test_empty_tokens(self):
        result = decode_tokens([], self.vocab, 2, self.special_ids)
        self.assertEqual(result, "")

