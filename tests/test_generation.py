
import unittest
from unittest.mock import patch

import torch

from src.models.translator import ISLTranslator


class TestGeneration(unittest.TestCase):

    def setUp(self):
        self.model = ISLTranslator(vocab_size=20)
        self.model.eval()

        self.pose = torch.randn(2, 16, 75, 3)
        self.mask = torch.ones(2, 16, 75, dtype=torch.bool)
        self.lengths = torch.tensor([16, 12])

    def test_stops_at_eos(self):
        # Force EOS to have the highest score.
        with torch.no_grad():
            self.model.decoder.output_layer.weight.zero_()
            self.model.decoder.output_layer.bias.zero_()
            self.model.decoder.output_layer.bias[2] = 10

        predictions = self.model.generate(self.pose, self.mask, self.lengths, max_length=5)

        self.assertEqual(predictions.shape, (2, 1))
        self.assertTrue(torch.all(predictions == 2).item())

    def test_max_length(self):
        # Force token 4 so EOS is never predicted.
        with torch.no_grad():
            self.model.decoder.output_layer.weight.zero_()
            self.model.decoder.output_layer.bias.zero_()
            self.model.decoder.output_layer.bias[4] = 10

        predictions = self.model.generate(self.pose, self.mask, self.lengths, max_length=3)

        self.assertEqual(predictions.shape, (2, 3))
        self.assertTrue(torch.all(predictions == 4).item())

    def test_pads_finished_sequences(self):
        step = [0]

        def fake_decode_step(encoded, keys, source_mask, hidden, current):
            logits = torch.zeros(2, 20, device=encoded.device)

            if step[0] == 0:
                logits[0, 2] = 10
                logits[1, 4] = 10
            else:
                logits[:, 2] = 10

            step[0] += 1
            return logits, hidden, torch.zeros(2, 16, device=encoded.device)

        with patch.object(self.model.decoder, "decode_step", side_effect=fake_decode_step):
            predictions = self.model.generate(self.pose, self.mask, self.lengths, max_length=5)

        self.assertEqual(predictions.tolist(), [[2, 0], [4, 2]])


if __name__ == "__main__":
    unittest.main()
