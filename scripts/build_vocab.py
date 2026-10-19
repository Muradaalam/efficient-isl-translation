
import csv
from pathlib import Path

from src.data.vocab import build_vocab, save_vocab, tokenize


MANIFEST_PATH = Path("experiments/isl_manifest_split.csv")
VOCAB_PATH = Path("experiments/vocab.json")


def main():
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    train_sentences = [row["sentence"] for row in rows if row["split"] == "train" and row["status"] == "valid"]

    if not train_sentences:
        raise ValueError("No valid training sentences found")

    vocab = build_vocab(train_sentences)
    save_vocab(vocab, VOCAB_PATH)

    print("Training recordings:", len(train_sentences))
    print("Unique training sentences:", len(set(train_sentences)))
    print("Vocabulary size:", len(vocab))
    print("Saved vocabulary to:", VOCAB_PATH)

    for split in ("val", "test"):
        sentences = [row["sentence"] for row in rows if row["split"] == split and row["status"] == "valid"]
        words = [word for sentence in sentences for word in tokenize(sentence)]
        unseen = sorted(set(word for word in words if word not in vocab))
        oov_rate = 100 * sum(word not in vocab for word in words) / len(words) if words else 0

        print(f"{split} unseen words:", unseen)
        print(f"{split} OOV rate: {oov_rate:.2f}%")


if __name__ == "__main__":
    main()
