
import os
import tempfile
import unittest

from src.data.vocab import build_vocab, encode, decode, save_vocab, load_vocab


class TestVocabulary(unittest.TestCase):

    def setUp(self):
        self.sentences = ["I go home", "I go school", "You go home"]
        self.vocab = build_vocab(self.sentences)

    def test_special_tokens(self):
        self.assertEqual(self.vocab["<PAD>"], 0)
        self.assertEqual(self.vocab["<SOS>"], 1)
        self.assertEqual(self.vocab["<EOS>"], 2)
        self.assertEqual(self.vocab["<UNK>"], 3)

    def test_encode_decode(self):
        tokens = encode("I go home", self.vocab)

        self.assertEqual(tokens[0], 1)
        self.assertEqual(tokens[-1], 2)
        self.assertEqual(decode(tokens, self.vocab), "i go home")

    def test_unknown_word(self):
        tokens = encode("I go moon", self.vocab)

        self.assertEqual(tokens[-2], 3)

    def test_vocab_save_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "vocab.json")
            save_vocab(self.vocab, path)
            loaded = load_vocab(path)

            self.assertEqual(loaded, self.vocab)

    def test_deterministic_ids(self):
        reversed_vocab = build_vocab(reversed(self.sentences))

        self.assertEqual(reversed_vocab, self.vocab)


if __name__ == "__main__":
    unittest.main()
