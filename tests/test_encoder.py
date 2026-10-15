
import unittest
import torch

from src.models.translator import PoseGRUEncoder


class TestPoseGRUEncoder(unittest.TestCase):

    def test_output_shapes(self):
        model = PoseGRUEncoder()

        pose = torch.randn(2, 64, 75, 3)
        mask = torch.ones(2, 64, 75, dtype=torch.bool)
        lengths = torch.tensor([64, 40])

        encoded, hidden = model(pose, mask, lengths)

        self.assertEqual(encoded.shape, (2, 64, 256))
        self.assertEqual(hidden.shape, (2, 2, 128))

    def test_invalid_length(self):
        model = PoseGRUEncoder()

        pose = torch.randn(1, 10, 75, 3)
        mask = torch.ones(1, 10, 75, dtype=torch.bool)
        lengths = torch.tensor([0])

        with self.assertRaises(ValueError):
            model(pose, mask, lengths)

    def test_backward_pass(self):
        model = PoseGRUEncoder()

        pose = torch.randn(2, 16, 75, 3)
        mask = torch.ones(2, 16, 75, dtype=torch.bool)
        lengths = torch.tensor([16, 12])

        encoded, _ = model(pose, mask, lengths)
        loss = encoded.sum()
        loss.backward()

        self.assertIsNotNone(model.gru.weight_ih_l0.grad)


if __name__ == "__main__":
    unittest.main()
