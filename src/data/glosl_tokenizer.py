
import sentencepiece as spm


class GloSLTokenizer:
    def __init__(self, model_path="experiments/glosl_spm_3000.model"):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.pad_id = self.sp.pad_id()
        self.sos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.unk_id = self.sp.unk_id()

    def __len__(self):
        return self.sp.vocab_size()

    def encode(self, sentence):
        sentence = " ".join(sentence.split())
        return [self.sos_id] + self.sp.encode(sentence, out_type=int) + [self.eos_id]

    def decode(self, token_ids):
        ids = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id == self.eos_id:
                break
            if token_id not in (self.pad_id, self.sos_id):
                ids.append(token_id)
        return self.sp.decode(ids)
