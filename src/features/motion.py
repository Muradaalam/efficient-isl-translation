
import numpy as np


def calculate_hand_motion(pose, valid_mask, return_counts=False):
    pose = np.asarray(pose)
    valid_mask = np.asarray(valid_mask)

    if pose.ndim != 3 or pose.shape[1:] != (75, 3):
        raise ValueError("Expected pose shape (T, 75, 3)")

    if valid_mask.shape != pose.shape[:2]:
        raise ValueError("Expected mask shape (T, 75)")

    T = len(pose)
    scores = np.zeros(T, dtype=np.float32)
    counts = np.zeros(T, dtype=np.int32)

    if T < 2:
        return (scores, counts) if return_counts else scores

    hands = pose[:, 33:75, :2]
    hand_mask = valid_mask[:, 33:75]

    valid_pairs = hand_mask[1:] & hand_mask[:-1]
    counts[1:] = valid_pairs.sum(axis=1)

    differences = hands[1:] - hands[:-1]
    distances = np.linalg.norm(differences, axis=-1)
    distances = np.where(valid_pairs, distances, 0)

    scores[1:] = distances.sum(axis=1) / np.maximum(counts[1:], 1)

    return (scores, counts) if return_counts else scores
