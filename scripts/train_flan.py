import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.data.flan_dataset import FLANGloSLDataset, FLANCollator
from src.models.flan_translator import FLANISLTranslator
from src.training.flan_checkpoint import save_checkpoint


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    return {
        key: value.to(device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


def train_epoch(model, loader, optimizer, scheduler, device, accum):
    model.train()

    total_loss = 0
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc="Training")

    for step, batch in enumerate(progress, 1):
        batch = move_batch(batch, device)

        outputs = model(**batch)

        loss = outputs.loss

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")

        group_start = ((step - 1) // accum) * accum
        group_size = min(accum, len(loader) - group_start)

        (loss / group_size).backward()

        if step % accum == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()

        progress.set_postfix(loss=total_loss / step)

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()

    total_loss = 0

    for batch in tqdm(loader, desc="Validation"):
        batch = move_batch(batch, device)

        outputs = model(**batch)

        total_loss += outputs.loss.item()

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pose-root", required=True)

    parser.add_argument("--pose-column", default="pose_file")
    parser.add_argument("--text-column", default="text")

    parser.add_argument(
        "--method",
        choices=["uniform", "motion-aware", "adaptive"],
        required=True
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--smoke-test", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    model_name = "google/flan-t5-xl"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    collator = FLANCollator(tokenizer)

    common = {
        "manifest": args.manifest,
        "pose_root": args.pose_root,
        "method": args.method,
        "target_frames": 128,
        "pose_column": args.pose_column,
        "text_column": args.text_column
    }

    train_data = FLANGloSLDataset(split="train", **common)
    val_data = FLANGloSLDataset(split="val", **common)

    print("Training samples:", len(train_data))
    print("Validation samples:", len(val_data))

    if args.smoke_test:
        train_data = torch.utils.data.Subset(train_data, range(2))
        val_data = torch.utils.data.Subset(val_data, range(2))
        args.epochs = 1

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.workers
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.workers
    )

    model = FLANISLTranslator(dtype=dtype).to(device)

    total = sum(p.numel() for p in model.parameters())

    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("Total parameters:", total)
    print("Trainable parameters:", trainable)

    optimizer = torch.optim.AdamW([
        p for p in model.parameters() if p.requires_grad
    ], lr=args.lr)

    steps = (
        (len(train_loader) + args.accum - 1) // args.accum
    ) * args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(steps * 0.05),
        num_training_steps=steps
    )

    output_dir = Path("experiments/flan_xl") / args.method

    if args.smoke_test:
        output_dir = output_dir / "smoke"

    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):

        train_loss = train_epoch(
            model, train_loader, optimizer,
            scheduler, device, args.accum
        )

        val_loss = validate(model, val_loader, device)

        result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        }

        history.append(result)

        print(result)

        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=4)

        if val_loss < best_loss:
            best_loss = val_loss
            stale_epochs = 0

            save_checkpoint(
                model,
                tokenizer,
                output_dir / "best",
                metadata={
                    "epoch": epoch,
                    "method": args.method,
                    "target_frames": 128,
                    "val_loss": val_loss
                }
            )

            print("Best checkpoint saved")

        else:
            stale_epochs += 1

        if stale_epochs >= args.patience:
            print("Early stopping")
            break


if __name__ == "__main__":
    main()