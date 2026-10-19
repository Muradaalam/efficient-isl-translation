
import unittest

import torch

from src.models.translator import ISLTranslator
from src.training.train import run_epoch


class TestTraining(unittest.TestCase):

    def test_weights_update(self):
        torch.manual_seed(42)

        model = ISLTranslator(vocab_size=20)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        batch = {
            "pose": torch.randn(2, 16, 75, 3),
            "valid_mask": torch.ones(2, 16, 75, dtype=torch.bool),
            "lengths": torch.tensor([16, 12]),
            "tokens": torch.tensor([[1, 4, 5, 2, 0], [1, 6, 2, 0, 0]])
        }

        original_weights = model.decoder.output_layer.weight.detach().clone()

        loss = run_epoch(model, [batch], optimizer)

        updated_weights = model.decoder.output_layer.weight.detach()

        self.assertTrue(torch.isfinite(torch.tensor(loss)).item())
        self.assertGreater(loss, 0)
        self.assertFalse(torch.equal(original_weights, updated_weights))

    def test_validation_does_not_update_weights(self):
        model = ISLTranslator(vocab_size=20)

        batch = {
            "pose": torch.randn(2, 16, 75, 3),
            "valid_mask": torch.ones(2, 16, 75, dtype=torch.bool),
            "lengths": torch.tensor([16, 12]),
            "tokens": torch.tensor([[1, 4, 5, 2, 0], [1, 6, 2, 0, 0]])
        }

        original_weights = model.decoder.output_layer.weight.detach().clone()

        loss = run_epoch(model, [batch])

        self.assertTrue(torch.isfinite(torch.tensor(loss)).item())
        self.assertTrue(torch.equal(original_weights, model.decoder.output_layer.weight))


if __name__ == "__main__":
    unittest.main()
