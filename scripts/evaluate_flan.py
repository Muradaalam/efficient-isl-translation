import argparse
import json
from pathlib import Path

import torch
import sacrebleu

from jiwer import wer
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sacrebleu.metrics import BLEU

from src.data.flan_dataset import FLANGloSLDataset, FLANCollator
from src.models.flan_translator import FLANISLTranslator
from src.training.flan_checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pose-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pose-column", default="pose_file")
    parser.add_argument("--text-column", default="text")

    parser.add_argument(
        "--method",
        choices=["uniform", "motion-aware", "adaptive"],
        required=True
    )

    parser.add_argument("--batch-size", type=int, default=2)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    checkpoint = Path(args.checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint / "tokenizer"
    )

    dataset = FLANGloSLDataset(
        args.manifest,
        args.pose_root,
        split="test",
        method=args.method,
        target_frames=128,
        pose_column=args.pose_column,
        text_column=args.text_column
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=FLANCollator(tokenizer)
    )

    model = FLANISLTranslator(dtype=dtype)

    model = load_checkpoint(model, checkpoint)

    model.to(device)
    model.eval()

    predictions = []
    references = []

    for batch in tqdm(loader, desc="Evaluating"):
        generated = model.generate(
            batch["pose"].to(device),
            batch["valid_mask"].to(device),
            batch["frame_mask"].to(device)
        )

        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True
        )

        predictions.extend(decoded)
        references.extend(batch["sentences"])

    predictions = [p.strip() for p in predictions]
    references = [r.strip() for r in references]

    results = {
        f"BLEU-{n}": BLEU(
            max_ngram_order=n,
            effective_order=True
        ).corpus_score(predictions, [references]).score
        for n in range(1, 5)
    }

    results["WER"] = wer(references, predictions) * 100

    results["Exact Match"] = 100 * sum(
        p == r for p, r in zip(predictions, references)
    ) / len(references)

    print(json.dumps(results, indent=4))

    with open(checkpoint.parent / "test_results.json", "w") as f:
        json.dump(results, f, indent=4)

    with open(checkpoint.parent / "predictions.json", "w") as f:
        json.dump([
            {"reference": r, "prediction": p}
            for r, p in zip(references, predictions)
        ], f, indent=4)


if __name__ == "__main__":
    main()