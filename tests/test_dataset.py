
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ISLPoseDataset, collate_fn
from src.data.vocab import build_vocab


class TestISLPoseDataset(unittest.TestCase):

    def test_dataset_and_batching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_dir = root / "features"
            feature_dir.mkdir()

            sentences = ["I go home", "You go to school"]
            vocab = build_vocab(sentences)

            rows = []

            for index, (frames, sentence) in enumerate(zip([20, 100], sentences)):
                video_name = f"video{index}.MP4"
                feature_path = feature_dir / f"video{index}.npz"

                pose = np.random.randn(frames, 75, 3).astype(np.float32)
                mask = np.ones((frames, 75), dtype=bool)

                np.savez_compressed(feature_path, pose=pose, valid_mask=mask)

                rows.append({
                    "sample_id": f"sample{index}",
                    "video_path": f"Videos_Sentence_Level/{video_name}",
                    "sentence": sentence,
                    "gloss": "",
                    "signer_id": "",
                    "split": "train",
                    "status": "valid"
                })

            manifest_path = root / "manifest.csv"

            with open(manifest_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            dataset = ISLPoseDataset(manifest_path, feature_dir, vocab)
            loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

            batch = next(iter(loader))

            self.assertEqual(len(dataset), 2)
            self.assertEqual(batch["pose"].shape, (2, 64, 75, 3))
            self.assertEqual(batch["valid_mask"].shape, (2, 64, 75))
            self.assertEqual(batch["lengths"].tolist(), [20, 64])
            self.assertEqual(batch["tokens"].shape[0], 2)

            self.assertTrue(torch.all(batch["pose"][0, 20:] == 0).item())
            self.assertTrue(torch.all(~batch["valid_mask"][0, 20:]).item())
            self.assertTrue(torch.all(batch["tokens"][:, 0] == 1).item())


if __name__ == "__main__":
    unittest.main()
