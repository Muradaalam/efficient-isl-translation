
import unittest

import torch
import torch.nn as nn

from src.models.translator import ISLTranslator


class TestISLTranslator(unittest.TestCase):

    def test_forward_pass(self):
        model = ISLTranslator(vocab_size=20)

        pose = torch.randn(2, 16, 75, 3)
        mask = torch.ones(2, 16, 75, dtype=torch.bool)
        lengths = torch.tensor([16, 12])

        tokens = torch.tensor([
            [1, 4, 5, 2, 0],
            [1, 6, 2, 0, 0],
        ])

        logits, attention = model(pose, mask, lengths, tokens)

        self.assertEqual(logits.shape, (2, 4, 20))
        self.assertEqual(attention.shape, (2, 4, 16))

    def test_backward_pass(self):
        model = ISLTranslator(vocab_size=20)

        pose = torch.randn(2, 16, 75, 3)
        mask = torch.ones(2, 16, 75, dtype=torch.bool)
        lengths = torch.tensor([16, 12])

        tokens = torch.tensor([
            [1, 4, 5, 2, 0],
            [1, 6, 2, 0, 0],
        ])

        logits, _ = model(pose, mask, lengths, tokens)

        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        loss = loss_fn(logits.reshape(-1, 20), tokens[:, 1:].reshape(-1))

        loss.backward()

        self.assertIsNotNone(model.encoder.gru.weight_ih_l0.grad)
        self.assertIsNotNone(model.decoder.gru.weight_ih_l0.grad)


if __name__ == "__main__":
    unittest.main()
