from pathlib import Path

import torch
from peft import PeftModel


def save_checkpoint(model, tokenizer, output_dir, metadata=None):
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    model.t5.save_pretrained(output_dir / "adapter")

    tokenizer.save_pretrained(output_dir / "tokenizer")

    torch.save({
        "projection": model.pose_projection.state_dict(),
        "normalization": model.pose_norm.state_dict(),
        "metadata": metadata or {}
    }, output_dir / "pose_layers.pt")


def load_checkpoint(model, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)

    state = torch.load(
        checkpoint_dir / "pose_layers.pt",
        map_location="cpu",
        weights_only=True
    )

    model.pose_projection.load_state_dict(state["projection"])

    model.pose_norm.load_state_dict(state["normalization"])

    base_model = model.t5.unload()

    model.t5 = PeftModel.from_pretrained(
        base_model,
        checkpoint_dir / "adapter"
    )

    return model