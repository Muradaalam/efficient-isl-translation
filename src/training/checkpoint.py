
import torch

from src.models.translator import ISLTranslator
from src.data.vocab import SPECIAL_TOKENS


def validate_vocab(vocab):
    for token, token_id in SPECIAL_TOKENS.items():
        if vocab.get(token) != token_id:
            raise ValueError(f"Invalid vocabulary mapping for {token}")

    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("Vocabulary IDs must be unique and consecutive")


def save_checkpoint(path, model, vocab, optimizer=None, epoch=0):
    validate_vocab(vocab)

    if len(vocab) != model.decoder.embedding.num_embeddings:
        raise ValueError("Vocabulary size does not match model")

    if vocab["<PAD>"] != model.decoder.embedding.padding_idx:
        raise ValueError("PAD ID does not match model")

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "model_config": {
            "vocab_size": len(vocab),
            "pad_token_id": vocab["<PAD>"]
        },
        "vocab": vocab,
        "special_tokens": SPECIAL_TOKENS.copy()
    }

    torch.save(checkpoint, path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    vocab = checkpoint["vocab"]
    validate_vocab(vocab)

    config = checkpoint["model_config"]

    if config["vocab_size"] != len(vocab) or config["pad_token_id"] != vocab["<PAD>"]:
        raise ValueError("Checkpoint model configuration does not match vocabulary")

    model = ISLTranslator(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, vocab, checkpoint
