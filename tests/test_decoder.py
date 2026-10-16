
import unittest

import torch
import torch.nn as nn

from src.models.translator import (PoseGRUEncoder,AttentionDecoder)


class TestAttentionDecoder(unittest.TestCase):

    def setUp(self):
        self.encoder = PoseGRUEncoder()
        self.decoder = AttentionDecoder(vocab_size=20)

        self.pose = torch.randn(2, 16, 75, 3)
        self.mask = torch.ones(2, 16, 75, dtype=torch.bool)
        self.lengths = torch.tensor([16, 12])

        # Example token IDs:
        # 0 = PAD, 1 = BOS, 2 = EOS
        self.tokens = torch.tensor([
            [1, 4, 5, 2, 0],
            [1, 6, 2, 0, 0],
        ])

    def test_output_shapes(self):
        encoded, hidden = self.encoder(self.pose, self.mask, self.lengths)

        logits, attention = self.decoder(encoded, hidden, self.lengths, self.tokens)

        self.assertEqual(logits.shape, (2, 4, 20))
        self.assertEqual(attention.shape, (2, 4, 16))

    def test_padding_attention(self):
        encoded, hidden = self.encoder(self.pose, self.mask, self.lengths)

        _, attention = self.decoder(encoded, hidden, self.lengths, self.tokens)

        # Video 2 has only 12 valid frames.
        # Frames 12, 13, 14, 15 must receive zero attention.
        self.assertTrue(torch.allclose(attention[1, :, 12:],torch.zeros_like(attention[1, :, 12:])))

    def test_backward_pass(self):
        encoded, hidden = self.encoder(self.pose, self.mask, self.lengths)

        logits, _ = self.decoder(encoded, hidden, self.lengths, self.tokens)

        loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        loss = loss_fn(logits.reshape(-1, 20),self.tokens[:, 1:].reshape(-1))

        loss.backward()

        self.assertIsNotNone(self.encoder.gru.weight_ih_l0.grad)
        self.assertIsNotNone(self.decoder.gru.weight_ih_l0.grad)


if __name__ == "__main__":
    unittest.main()
