
import json
from collections import Counter


PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3

SPECIAL_TOKENS = {
    "<PAD>": PAD_ID,
    "<SOS>": SOS_ID,
    "<EOS>": EOS_ID,
    "<UNK>": UNK_ID
}


def tokenize(sentence):
    return sentence.lower().split()


def build_vocab(sentences, min_freq=1):
    if min_freq < 1:
        raise ValueError("min_freq must be at least 1")

    counts = Counter()

    for sentence in sentences:
        counts.update(tokenize(sentence))

    vocab = SPECIAL_TOKENS.copy()

    for word in sorted(counts, key=lambda word: (-counts[word], word)):
        if counts[word] >= min_freq and word not in vocab:
            vocab[word] = len(vocab)

    return vocab


def encode(sentence, vocab):
    token_ids = [SOS_ID]

    for word in tokenize(sentence):
        token_ids.append(vocab.get(word, UNK_ID))

    token_ids.append(EOS_ID)
    return token_ids


def decode(token_ids, vocab):
    id_to_word = {token_id: word for word, token_id in vocab.items()}
    words = []

    for token_id in token_ids:
        token_id = int(token_id)

        if token_id == EOS_ID:
            break

        if token_id in (PAD_ID, SOS_ID):
            continue

        words.append(id_to_word.get(token_id, "<UNK>"))

    return " ".join(words)


def save_vocab(vocab, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(vocab, file, indent=2, ensure_ascii=False)


def load_vocab(path):
    with open(path, "r", encoding="utf-8") as file:
        vocab = json.load(file)

    for token, token_id in SPECIAL_TOKENS.items():
        if vocab.get(token) != token_id:
            raise ValueError(f"Invalid vocabulary mapping for {token}")

    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("Vocabulary IDs must be unique and consecutive")

    return vocab
