import numpy as np


TARGET_FRAMES = 128
HAND_START = 33
HAND_END = 75


def motion_scores(pose, valid_mask):
    hands = pose[:, HAND_START:HAND_END]
    valid = valid_mask[:, HAND_START:HAND_END]

    scores = np.zeros(len(pose), dtype=np.float32)

    if len(pose) < 2:
        return scores

    displacement = np.linalg.norm(
        hands[1:] - hands[:-1], axis=-1
    )

    pair_valid = valid[1:] & valid[:-1]
    displacement = np.where(pair_valid, displacement, 0)

    counts = pair_valid.sum(axis=1)

    scores[1:] = displacement.sum(axis=1) / np.maximum(counts, 1)

    return scores


def uniform_indices(n, target):
    if n <= target:
        return np.arange(n)

    return np.linspace(0, n - 1, target).round().astype(int)


def motion_aware_indices(pose, valid_mask, target):
    scores = motion_scores(pose, valid_mask)

    if np.allclose(scores, 0):
        return uniform_indices(len(pose), target)

    indices = np.argsort(-scores, kind="stable")[:target]

    return np.sort(indices)


def adaptive_indices(pose, valid_mask, target):
    n = len(pose)

    if n <= target:
        return np.arange(n)

    uniform_count = target // 2
    uniform = uniform_indices(n, uniform_count)

    scores = motion_scores(pose, valid_mask)

    remaining = np.setdiff1d(np.arange(n), uniform)

    if np.allclose(scores, 0):
        return uniform_indices(n, target)

    ranked = remaining[
        np.argsort(-scores[remaining], kind="stable")
    ]

    selected = np.concatenate([
        uniform,
        ranked[:target - len(uniform)]
    ])

    return np.sort(selected)


def sample(pose, valid_mask, target_frames=128, method="uniform"):
    if len(pose) == 0:
        raise ValueError("Empty pose sequence")

    if len(pose) <= target_frames:
        indices = np.arange(len(pose))

    elif method == "uniform":
        indices = uniform_indices(len(pose), target_frames)

    elif method == "motion-aware":
        indices = motion_aware_indices(
            pose, valid_mask, target_frames
        )

    elif method == "adaptive":
        indices = adaptive_indices(
            pose, valid_mask, target_frames
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    sampled_pose = pose[indices]
    sampled_valid = valid_mask[indices]

    frame_mask = sampled_valid.any(axis=1)

    padding = target_frames - len(indices)

    if padding > 0:
        sampled_pose = np.pad(
            sampled_pose, ((0, padding), (0, 0), (0, 0))
        )

        sampled_valid = np.pad(
            sampled_valid, ((0, padding), (0, 0))
        )

        frame_mask = np.pad(frame_mask, (0, padding))
        indices = np.pad(indices, (0, padding), constant_values=-1)

    return sampled_pose, sampled_valid, frame_mask, indices