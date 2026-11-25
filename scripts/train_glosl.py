

import argparse
import csv
import importlib
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data.dataset import collate_fn
from src.data.glosl_dataset import GloSLPoseDataset
from src.data.glosl_tokenizer import GloSLTokenizer
from src.models.translator import ISLTranslator
from src.training.glosl_checkpoint import save_glosl_checkpoint


def get_args():
    parser = argparse.ArgumentParser(description="Train GloSL Uniform-64 or Adaptive-64 GRU")
    parser.add_argument("--manifest", type=Path, default=Path("../glosl_8000/splits/shared_splits.csv"))
    parser.add_argument("--pose-root", type=Path, default=Path("../glosl_8000"))
    parser.add_argument("--tokenizer", type=Path, default=Path("experiments/glosl_spm_3000.model"))
    parser.add_argument("--pose-loader", default="auto", help="MODULE:FUNCTION, or auto")
    parser.add_argument("--method", required=True, choices=["uniform", "adaptive"])
    parser.add_argument("--target-frames", type=int, default=64)
    parser.add_argument("--num-landmarks", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--save-dir", type=Path, default=Path("experiments/checkpoints"))
    parser.add_argument("--resume", type=Path, default=None, help="Resume from the matching _last.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--patience", type=int, default=0, help="Early-stop patience; 0 disables")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def get_pose_loader(spec):
    if spec == "auto":
        candidates = [
            "src.features.glosl_pose:read_glosl_pose",
            "src.features.pose_motion:read_glosl_pose",
        ]
    else:
        candidates = [spec]

    for candidate in candidates:
        module_name, sep, function_name = candidate.partition(":")
        if not sep:
            raise ValueError("--pose-loader must be MODULE:FUNCTION")
        try:
            module = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError):
            if spec != "auto":
                raise
            continue
        loader = getattr(module, function_name, None)
        if callable(loader):
            print(f"Pose loader: {candidate}", flush=True)
            return loader
    raise RuntimeError("Pose loader not found. Specify --pose-loader MODULE:FUNCTION")


def make_dataloader(manifest, pose_root, tokenizer, pose_loader, split, method,
                    frames, batch_size, num_workers, seed, limit, device):
    dataset = GloSLPoseDataset(
        manifest_path=manifest,
        pose_root=pose_root,
        tokenizer=tokenizer,
        pose_loader=pose_loader,
        split=split,
        method=method,
        target_frames=frames,
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("Sample limits must be positive")
        dataset = Subset(dataset, range(min(limit, len(dataset))))

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == "train"),
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"), generator=generator,
    )


def run_epoch(model, loader, optimizer, device, pad_id, grad_clip, log_every):
    training = optimizer is not None
    model.train(training)
    total_loss, total_tokens = 0.0, 0

    for batch_index, batch in enumerate(loader, start=1):
        pose = batch["pose"].to(device, dtype=torch.float32, non_blocking=True)
        mask = batch["valid_mask"].to(device, dtype=torch.bool, non_blocking=True)
        lengths = batch["lengths"].to(device, dtype=torch.long, non_blocking=True)
        tokens = batch["tokens"].to(device, dtype=torch.long, non_blocking=True)

        if pose.shape[2:] != (model.encoder.num_landmarks, 3):
            raise ValueError(f"Unexpected pose shape: {tuple(pose.shape)}")
        if tokens.shape[1] < 2:
            raise ValueError("Target sequences must contain SOS and EOS")

        with torch.set_grad_enabled(training):
            logits, _ = model(pose, mask, lengths, tokens)
            targets = tokens[:, 1:]
            if logits.shape[:2] != targets.shape:
                raise ValueError(f"Logits {tuple(logits.shape)} and targets {tuple(targets.shape)} disagree")
            n_tokens = int((targets != pad_id).sum().item())
            if n_tokens == 0:
                continue
            loss_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       targets.reshape(-1), ignore_index=pad_id, reduction="sum")
            loss = loss_sum / n_tokens
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at batch {batch_index}")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        total_loss += loss_sum.detach().item()
        total_tokens += n_tokens
        if training and log_every > 0 and batch_index % log_every == 0:
            print(f"  batch {batch_index}/{len(loader)} | token loss {total_loss / total_tokens:.4f}", flush=True)

    if total_tokens == 0:
        raise RuntimeError("No non-PAD target tokens found")
    return total_loss / total_tokens


def read_resume(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model_state = checkpoint.get("model_state_dict", checkpoint.get("model_state"))
    optimizer_state = checkpoint.get("optimizer_state_dict", checkpoint.get("optimizer_state"))
    if model_state is None or optimizer_state is None:
        raise ValueError("Resume checkpoint must contain model and optimizer states")
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    epoch = int(checkpoint.get("epoch", 0))
    print(f"Resumed from epoch {epoch}: {path}", flush=True)
    return epoch


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_history(path, history):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "seconds"])
        writer.writeheader()
        writer.writerows(history)


def main():
    args = get_args()
    if args.epochs < 1 or args.batch_size < 1 or args.target_frames < 1 or args.num_workers < 0:
        raise ValueError("epochs, batch-size and target-frames must be positive; num-workers >= 0")
    if args.lr <= 0 or args.grad_clip <= 0 or args.weight_decay < 0:
        raise ValueError("lr and grad-clip must be positive, weight-decay must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          "cpu" if args.device == "auto" else args.device)
    for file_path in (args.manifest, args.tokenizer):
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
    if not args.pose_root.exists():
        raise FileNotFoundError(args.pose_root)

    tokenizer = GloSLTokenizer(str(args.tokenizer))
    pose_loader = get_pose_loader(args.pose_loader)
    train_loader = make_dataloader(args.manifest, args.pose_root, tokenizer, pose_loader, "train",
                                   args.method, args.target_frames, args.batch_size, args.num_workers,
                                   args.seed, args.train_limit, device)
    val_loader = make_dataloader(args.manifest, args.pose_root, tokenizer, pose_loader, "val",
                                 args.method, args.target_frames, args.batch_size, args.num_workers,
                                 args.seed, args.val_limit, device)

    model = ISLTranslator(vocab_size=len(tokenizer), pad_token_id=0,
                          num_landmarks=args.num_landmarks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | Method: {args.method} | Parameters: {n_params:,}", flush=True)
    print(f"Train: {len(train_loader.dataset)} | Validation: {len(val_loader.dataset)} | Vocab: {len(tokenizer)}", flush=True)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"glosl_{args.method}"
    best_path = args.save_dir / f"{prefix}_best.pt"
    last_path = args.save_dir / f"{prefix}_last.pt"
    history_path = args.save_dir / f"{prefix}_history.csv"
    config_path = args.save_dir / f"{prefix}_config.json"
    current_config = {
        "method": args.method, "manifest": str(args.manifest.resolve()),
        "pose_root": str(args.pose_root.resolve()), "tokenizer": str(args.tokenizer.resolve()),
        "pose_loader": args.pose_loader, "target_frames": args.target_frames,
        "tokenizer_sha256": file_sha256(args.tokenizer),
        "num_landmarks": args.num_landmarks, "vocab_size": len(tokenizer),
        "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip, "seed": args.seed, "train_limit": args.train_limit,
        "val_limit": args.val_limit, "device": str(device), "parameters": n_params,
    }

    start_epoch = 0
    history = []
    best_val = float("inf")
    if args.resume is not None:
        if config_path.is_file():
            prior = json.loads(config_path.read_text(encoding="utf-8"))
            checks = ("method", "target_frames", "num_landmarks", "vocab_size", "batch_size",
                      "lr", "weight_decay", "train_limit", "val_limit", "tokenizer_sha256", "grad_clip", "seed")
            for key in checks:
                if prior.get(key) != current_config[key]:
                    raise ValueError(f"Resume configuration differs: {key}")
        start_epoch = read_resume(args.resume, model, optimizer, device)
        if history_path.is_file():
            with open(history_path, newline="", encoding="utf-8") as f:
                history = list(csv.DictReader(f))
            prior_rows = [r for r in history if int(r["epoch"]) <= start_epoch]
            history = prior_rows
            if prior_rows:
                best_val = min(float(r["val_loss"]) for r in prior_rows)
    else:
        for path in (best_path, last_path, history_path, config_path):
            if path.exists():
                raise FileExistsError(f"Output already exists: {path}. Use a new --save-dir or --resume")

    config_path.write_text(json.dumps(current_config, indent=2), encoding="utf-8")
    stale_epochs = 0
    for epoch in range(start_epoch + 1, args.epochs + 1):
        started = time.perf_counter()
        train_loss = run_epoch(model, train_loader, optimizer, device, 0, args.grad_clip, args.log_every)
        val_loss = run_epoch(model, val_loader, None, device, 0, args.grad_clip, 0)
        seconds = time.perf_counter() - started
        row = {"epoch": epoch, "train_loss": round(train_loss, 6),
               "val_loss": round(val_loss, 6), "seconds": round(seconds, 2)}
        history.append(row)
        save_history(history_path, history)
        save_glosl_checkpoint(str(last_path), model, str(args.tokenizer), optimizer=optimizer,
                              epoch=epoch, val_loss=val_loss)
        if val_loss < best_val:
            best_val = val_loss
            stale_epochs = 0
            save_glosl_checkpoint(str(best_path), model, str(args.tokenizer), optimizer=optimizer,
                                  epoch=epoch, val_loss=val_loss)
            label = "BEST"
        else:
            stale_epochs += 1
            label = ""
        print(f"Epoch {epoch}/{args.epochs} | train={train_loss:.4f} | val={val_loss:.4f} | "
              f"seconds={seconds:.1f} {label}", flush=True)
        if args.patience and stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} epochs without improvement", flush=True)
            break

    print(f"Best validation checkpoint: {best_path}", flush=True)
    print(f"Last checkpoint: {last_path}", flush=True)
    print(f"Training log: {history_path}", flush=True)


if __name__ == "__main__":
    main()
