
import hashlib
from pathlib import Path

import torch

from src.data.glosl_tokenizer import GloSLTokenizer
from src.models.translator import ISLTranslator


def tokenizer_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_glosl_checkpoint(path, model, tokenizer_path, optimizer=None, epoch=0, val_loss=None):
    tokenizer = GloSLTokenizer(tokenizer_path)

    if len(tokenizer) != model.decoder.embedding.num_embeddings:
        raise ValueError("Tokenizer vocabulary does not match model")

    if tokenizer.pad_id != model.decoder.embedding.padding_idx:
        raise ValueError("Tokenizer PAD ID does not match model")

    checkpoint = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "model_config": {
            "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_id,
            "num_landmarks": model.encoder.num_landmarks
        },
        "tokenizer_file": Path(tokenizer_path).name,
        "tokenizer_sha256": tokenizer_hash(tokenizer_path)
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_glosl_checkpoint(path, tokenizer_path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    tokenizer = GloSLTokenizer(tokenizer_path)

    if tokenizer_hash(tokenizer_path) != checkpoint["tokenizer_sha256"]:
        raise ValueError("Checkpoint tokenizer does not match supplied tokenizer")

    config = checkpoint["model_config"]

    if len(tokenizer) != config["vocab_size"] or tokenizer.pad_id != config["pad_token_id"]:
        raise ValueError("Invalid tokenizer configuration")

    model = ISLTranslator(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, tokenizer, checkpoint
