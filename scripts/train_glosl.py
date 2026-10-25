
import argparse
import importlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import collate_fn
from src.data.glosl_dataset import GloSLPoseDataset
from src.data.glosl_tokenizer import GloSLTokenizer
from src.models.translator import ISLTranslator
from src.training.train import run_epoch
from src.training.glosl_checkpoint import save_glosl_checkpoint


def load_function(name):
    module_name, function_name = name.rsplit(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def main():
    parser = argparse.ArgumentParser(description="Train GRU on GloSL-ISL")
    parser.add_argument("--pose-loader", required=True, help="Module:function from Shahjad")
    parser.add_argument("--method", choices=["uniform", "adaptive"], default="uniform")
    parser.add_argument("--manifest", default="../glosl_8000/splits/shared_splits.csv")
    parser.add_argument("--pose-root", default="../glosl_8000")
    parser.add_argument("--tokenizer", default="experiments/glosl_spm_3000.model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("Invalid training configuration")

    if args.weight_decay < 0 or args.train_limit < 0 or args.val_limit < 0:
        raise ValueError("Invalid training configuration")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = GloSLTokenizer(args.tokenizer)
    pose_loader = load_function(args.pose_loader)

    train_data = GloSLPoseDataset(
        args.manifest, args.pose_root, tokenizer, pose_loader,
        split="train", method=args.method, target_frames=64
    )
    val_data = GloSLPoseDataset(
        args.manifest, args.pose_root, tokenizer, pose_loader,
        split="val", method=args.method, target_frames=64
    )

    if args.train_limit:
        train_data.records = train_data.records[:args.train_limit]
    if args.val_limit:
        val_data.records = val_data.records[:args.val_limit]

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    model = ISLTranslator(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_id,
        num_landmarks=100
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    output = Path(f"experiments/checkpoints/glosl_{args.method}_best.pt")
    output.parent.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update({"seed": 42, "target_frames": 64, "num_landmarks": 100})
    output.with_suffix(".json").write_text(json.dumps(config, indent=2))

    print("Method:", args.method)
    print("Training samples:", len(train_data))
    print("Validation samples:", len(val_data))
    print("Vocabulary size:", len(tokenizer))
    print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss = run_epoch(
            model, train_loader, optimizer=optimizer,
            device=device, pad_id=tokenizer.pad_id
        )
        val_loss = run_epoch(
            model, val_loader, device=device, pad_id=tokenizer.pad_id
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"Time: {time.time() - start:.1f}s",
            flush=True
        )

        if val_loss < best_loss:
            best_loss = val_loss
            save_glosl_checkpoint(
                output, model, args.tokenizer,
                optimizer=optimizer, epoch=epoch, val_loss=val_loss
            )
            print("Saved best checkpoint:", output, flush=True)

    print("Best validation loss:", best_loss)


if __name__ == "__main__":
    main()
