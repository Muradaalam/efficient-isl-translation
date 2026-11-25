from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features.glosl_pose_full import read_glosl_pose_full
from src.sampling.temporal_128 import sample


class FLANGloSLDataset(Dataset):

    def __init__(
        self,
        manifest,
        pose_root,
        split="train",
        method="uniform",
        target_frames=128,
        pose_column="pose_file",
        text_column="text"
    ):
        df = pd.read_csv(manifest)

        df = df[df["split"] == split].reset_index(drop=True)

        if pose_column not in df.columns:
            raise ValueError(f"Missing column: {pose_column}")

        if text_column not in df.columns:
            raise ValueError(f"Missing column: {text_column}")

        self.df = df
        self.pose_root = Path(pose_root)
        self.method = method
        self.target_frames = target_frames
        self.pose_column = pose_column
        self.text_column = text_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = Path(str(row[self.pose_column]))

        if not path.is_absolute():
            path = self.pose_root / path

        if not path.exists():
            path = self.pose_root / "poses" / path.name

        pose, valid, _ = read_glosl_pose_full(path)

        pose, valid, frame_mask, indices = sample(
            pose,
            valid,
            target_frames=self.target_frames,
            method=self.method
        )

        if not frame_mask.any():
            raise ValueError(f"No valid frames: {path}")

        return {
            "pose": torch.from_numpy(pose).float(),
            "valid_mask": torch.from_numpy(valid).bool(),
            "frame_mask": torch.from_numpy(frame_mask).long(),
            "indices": torch.from_numpy(indices).long(),
            "sentence": str(row[self.text_column])
        }


class FLANCollator:

    def __init__(self, tokenizer, max_target_length=128):
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

    def __call__(self, batch):
        sentences = [item["sentence"] for item in batch]

        encoded = self.tokenizer(
            text_target=sentences,
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt"
        )

        labels = encoded["input_ids"].clone()

        labels[encoded["attention_mask"] == 0] = -100

        return {
            "pose": torch.stack([b["pose"] for b in batch]),
            "valid_mask": torch.stack([b["valid_mask"] for b in batch]),
            "frame_mask": torch.stack([b["frame_mask"] for b in batch]),
            "labels": labels,
            "sentences": sentences
        }