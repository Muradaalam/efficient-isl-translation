
import os
import tempfile
import unittest

import torch

from src.data.vocab import build_vocab
from src.models.translator import ISLTranslator
from src.training.checkpoint import save_checkpoint, load_checkpoint


class TestCheckpoint(unittest.TestCase):

    def test_save_load(self):
        torch.manual_seed(42)

        vocab = build_vocab(["I go home", "You go school"])
        model = ISLTranslator(vocab_size=len(vocab))
        model.eval()

        pose = torch.randn(2, 16, 75, 3)
        mask = torch.ones(2, 16, 75, dtype=torch.bool)
        lengths = torch.tensor([16, 12])
        tokens = torch.tensor([[1, 4, 5, 2], [1, 6, 2, 0]])

        with torch.no_grad():
            original_logits, _ = model(pose, mask, lengths, tokens)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pt")

            save_checkpoint(path, model, vocab, epoch=1)
            loaded_model, loaded_vocab, checkpoint = load_checkpoint(path)

            with torch.no_grad():
                loaded_logits, _ = loaded_model(pose, mask, lengths, tokens)

            self.assertEqual(loaded_vocab, vocab)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["model_config"]["vocab_size"], len(vocab))
            self.assertTrue(torch.allclose(original_logits, loaded_logits))


if __name__ == "__main__":
    unittest.main()
