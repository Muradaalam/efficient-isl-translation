
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.sampling.temporal import sample


class GloSLPoseDataset(Dataset):
    def __init__(self, manifest_path, pose_root, tokenizer, pose_loader, split="train", method="uniform", target_frames=64):
        if method not in ("uniform", "adaptive"):
            raise ValueError("method must be uniform or adaptive")

        if target_frames < 1:
            raise ValueError("target_frames must be positive")

        self.pose_root = Path(pose_root)
        self.tokenizer = tokenizer
        self.pose_loader = pose_loader
        self.method = method
        self.target_frames = target_frames

        with open(manifest_path, newline="", encoding="utf-8") as f:
            self.records = [row for row in csv.DictReader(f) if row["split"] == split]

        if not self.records:
            raise ValueError(f"No samples found for split '{split}'")

        for row in self.records:
            path = self.pose_root / row["pose_file"]
            if not path.is_file():
                raise FileNotFoundError(path)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        path = self.pose_root / row["pose_file"]

        pose, valid_mask = self.pose_loader(path)
        pose = np.asarray(pose, dtype=np.float32)
        valid_mask = np.asarray(valid_mask, dtype=bool)

        if pose.ndim != 3 or pose.shape[1:] != (100, 3):
            raise ValueError(f"Unexpected pose shape: {pose.shape}")

        if valid_mask.shape != pose.shape[:2]:
            raise ValueError("Invalid landmark mask shape")

        if not np.isfinite(pose).all():
            raise ValueError(f"Non-finite coordinates in {path}")

        sampled_pose, sampled_mask, indices = sample(
            pose, valid_mask, target_frames=self.target_frames, method=self.method
        )

        length = int(np.count_nonzero(indices >= 0))
        if length < 1:
            raise ValueError(f"No usable frames in {path}")

        tokens = self.tokenizer.encode(row["text"])

        return {
            "pose": torch.from_numpy(np.asarray(sampled_pose).copy()).float(),
            "valid_mask": torch.from_numpy(np.asarray(sampled_mask).copy()).bool(),
            "length": length,
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "sentence": row["text"],
            "sample_id": row["id"]
        }
