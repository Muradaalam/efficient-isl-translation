import torch
import unittest

def decode_tokens(token_ids, id_to_token, eos_id, special_ids):
    words = []

    for token_id in token_ids:
        token_id = int(token_id)

        if token_id == eos_id:
            break

        if token_id in special_ids:
            continue

        if token_id not in id_to_token:
            raise ValueError(f"Unknown token ID: {token_id}")

        words.append(id_to_token[token_id])

    return " ".join(words)

def predict_tokens(model, pose, valid_mask, lengths):
    if pose.ndim != 4:
        raise ValueError("Pose must have shape [B, T, J, 3]")

    if valid_mask.shape != pose.shape[:3]:
        raise ValueError("Invalid validity mask shape")

    if lengths.ndim != 1 or lengths.shape[0] != pose.shape[0]:
        raise ValueError("Invalid sequence lengths shape")

    if torch.any(lengths <= 0) or torch.any(lengths > pose.shape[1]):
        raise ValueError("Sequence lengths must be between 1 and T")

    model.eval()

    with torch.inference_mode():
        token_ids = model.predict(pose, valid_mask, lengths)

    return token_ids


class MockModel(torch.nn.Module):

    def predict(self, pose, valid_mask, lengths):
        return torch.tensor([[1, 3, 4, 5, 2]])


class TestPredictTokens(unittest.TestCase):

    def setUp(self):
        self.model = MockModel()
        self.pose = torch.zeros(1, 10, 75, 3)
        self.valid_mask = torch.ones(1, 10, 75, dtype=torch.bool)
        self.lengths = torch.tensor([10])

    def test_prediction(self):
        output = predict_tokens(self.model, self.pose, self.valid_mask, self.lengths)
        self.assertEqual(output.tolist(), [[1, 3, 4, 5, 2]])

    def test_invalid_pose_shape(self):
        with self.assertRaises(ValueError):
            predict_tokens(self.model, torch.zeros(10, 75, 3), self.valid_mask, self.lengths)

    def test_invalid_mask_shape(self):
        wrong_mask = torch.ones(1, 10, 74, dtype=torch.bool)

        with self.assertRaises(ValueError):
            predict_tokens(self.model, self.pose, wrong_mask, self.lengths)

    def test_invalid_lengths(self):
        with self.assertRaises(ValueError):
            predict_tokens(self.model, self.pose, self.valid_mask, torch.tensor([0]))


def predict_sentence(model, pose, valid_mask, lengths, id_to_token, eos_id, special_ids):
    token_ids = predict_tokens(model, pose, valid_mask, lengths)

    if token_ids.ndim != 2 or token_ids.shape[0] != pose.shape[0]:
        raise ValueError("Expected token IDs with shape [B, L]")

    sentences = []

    for sequence in token_ids:
        sentence = decode_tokens(sequence, id_to_token, eos_id, special_ids)
        sentences.append(sentence)

    return sentences

