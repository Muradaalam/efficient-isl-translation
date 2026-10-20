
import csv
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.data.vocab import encode
from src.sampling.temporal import sample


class ISLPoseDataset(Dataset):
    def __init__(self, manifest_path, feature_dir, vocab, split="train", target_frames=64, existing_only=False):
        self.feature_dir = Path(feature_dir)
        self.vocab = vocab
        self.target_frames = target_frames
        self.records = []

        with open(manifest_path, newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        for row in rows:
            if row["split"] != split or row["status"] != "valid":
                continue

            video_path = Path(row["video_path"])

            # Manifest includes Videos_Sentence_Level, but the feature folder starts inside it.
            if video_path.parts[0] == "Videos_Sentence_Level":
                video_path = Path(*video_path.parts[1:])

            feature_path = self.feature_dir / video_path.with_suffix(".npz")

            if existing_only and not feature_path.is_file():
                continue

            self.records.append({
                "sample_id": row["sample_id"],
                "feature_path": feature_path,
                "sentence": row["sentence"]
            })

        if not self.records:
            raise ValueError(f"No usable samples found for split '{split}'")

        if not existing_only:
            missing = [row["feature_path"] for row in self.records if not row["feature_path"].is_file()]

            if missing:
                raise FileNotFoundError(f"{len(missing)} feature files missing. Example: {missing[0]}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        with np.load(record["feature_path"]) as data:
            pose = data["pose"].astype(np.float32)
            valid_mask = data["valid_mask"].astype(bool)

        sampled_pose, sampled_mask, indices = sample(pose, valid_mask, target_frames=self.target_frames)

        length = int(np.sum(indices >= 0))
        tokens = encode(record["sentence"], self.vocab)

        return {
            "pose": torch.from_numpy(sampled_pose.copy()).float(),
            "valid_mask": torch.from_numpy(sampled_mask.copy()).bool(),
            "length": length,
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "sentence": record["sentence"]
        }


def collate_fn(samples):
    poses = torch.stack([item["pose"] for item in samples])
    masks = torch.stack([item["valid_mask"] for item in samples])
    lengths = torch.tensor([item["length"] for item in samples], dtype=torch.long)
    tokens = pad_sequence([item["tokens"] for item in samples], batch_first=True, padding_value=0)

    return {
        "pose": poses,
        "valid_mask": masks,
        "lengths": lengths,
        "tokens": tokens
    }
