import unittest
import torch

from src.inference.predict import decode_tokens, predict_tokens, predict_sentence


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

class MockModel(torch.nn.Module):

    def predict(self, pose, valid_mask, lengths):
        return torch.tensor([[1, 3, 4, 5, 2]])

class TestPredictSentence(unittest.TestCase):

    def setUp(self):
        self.model = MockModel()
        self.pose = torch.zeros(1, 10, 75, 3)
        self.valid_mask = torch.ones(1, 10, 75, dtype=torch.bool)
        self.lengths = torch.tensor([10])

        self.vocab = {
            0: "<pad>",
            1: "<sos>",
            2: "<eos>",
            3: "i",
            4: "need",
            5: "water"
        }

    def test_sentence_prediction(self):
        sentences = predict_sentence(
            self.model, self.pose, self.valid_mask, self.lengths,
            self.vocab, 2, {0, 1, 2}
        )

        self.assertEqual(sentences, ["i need water"])

    def test_invalid_model_output(self):
        class InvalidModel(torch.nn.Module):
            def predict(self, pose, valid_mask, lengths):
                return torch.tensor([1, 3, 4, 5, 2])

        with self.assertRaises(ValueError):
            predict_sentence(
                InvalidModel(), self.pose, self.valid_mask,
                self.lengths, self.vocab, 2, {0, 1, 2}
            )

    def test_batch_prediction(self):
        class BatchMockModel(torch.nn.Module):
            def predict(self, pose, valid_mask, lengths):
                return torch.tensor([
                    [1, 3, 4, 5, 2],
                    [1, 3, 4, 2, 0]
                ])

        pose = torch.zeros(2, 10, 75, 3)
        valid_mask = torch.ones(2, 10, 75, dtype=torch.bool)
        lengths = torch.tensor([10, 8])

        sentences = predict_sentence(
            BatchMockModel(), pose, valid_mask, lengths,
            self.vocab, 2, {0, 1, 2}
        )

        self.assertEqual(sentences, ["i need water", "i need"])

