
import numpy as np

# Landmark layout:
# 0–32   : Body (33)
# 33–74  : Hands (42)
# 75–99  : Face (25)

HAND_START = 33
HAND_END = 75  # Python slicing excludes index 75


def calculate_motion_scores(pose, valid_mask):
    """
    Calculate motion using ONLY the 42 hand landmarks.

    Input:
        pose: (T, 100, 3)
        valid_mask: (T, 100)

    Output:
        scores: (T,)
    """

    pose = np.asarray(pose)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    # Validate dimensions without hardcoding 75 landmarks
    if pose.ndim != 3 or pose.shape[2] != 3 or pose.shape[1] < HAND_END:
        raise ValueError("Expected pose shape (T, K, 3), K >= 75")

    if valid_mask.shape != pose.shape[:2]:
        raise ValueError("Invalid mask shape")

    if not np.isfinite(pose).all():
        raise ValueError("Pose coordinates must be finite")

    T = pose.shape[0]

    # Initialize every frame's motion score to zero
    scores = np.zeros(T, dtype=np.float64)

    if T < 2:
        return scores

    # Extract only hand landmarks (indices 33–74)
    hands = pose[:, HAND_START:HAND_END, :]
    hand_mask = valid_mask[:, HAND_START:HAND_END]

    # Difference between consecutive frames
    differences = hands[1:] - hands[:-1]

    # Euclidean distance for each hand landmark
    distances = np.linalg.norm(differences, axis=2)

    # Landmark must be valid in both consecutive frames
    valid_pairs = hand_mask[1:] & hand_mask[:-1]

    # Count valid hand landmarks
    counts = valid_pairs.sum(axis=1)

    # Average displacement, safely handling zero valid pairs
    scores[1:] = np.divide(
        (distances * valid_pairs).sum(axis=1),
        counts,
        out=np.zeros(T - 1, dtype=np.float64),
        where=counts > 0
    )

    return scores


def sample(pose, valid_mask, target_frames=64, method="uniform"):
    """
    Uniform or hybrid adaptive temporal sampling.

    Input:
        pose: (T, 100, 3)
        valid_mask: (T, 100)
        target_frames: 64 by default
        method: "uniform" or "adaptive"

    Returns:
        sampled_pose: (64, 100, 3)
        sampled_mask: (64, 100)
        indices: (64,)
    """

    pose = np.asarray(pose)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    # Step 1: Validate input
    if pose.ndim != 3 or pose.shape[2] != 3 or pose.shape[1] < HAND_END:
        raise ValueError("Expected pose shape (T, K, 3), K >= 75")

    if valid_mask.shape != pose.shape[:2]:
        raise ValueError("Invalid mask shape")

    if (
        not isinstance(target_frames, (int, np.integer))
        or isinstance(target_frames, (bool, np.bool_))
        or target_frames <= 0
    ):
        raise ValueError("target_frames must be a positive integer")

    if method not in ("uniform", "adaptive"):
        raise ValueError("Unsupported sampling method")

    T = pose.shape[0]

    if T == 0:
        raise ValueError("Cannot sample an empty video")

    n = min(T, target_frames)

    # Step 2: Uniform sampling
    if method == "uniform":

        indices = np.linspace(
            0, T - 1, n, dtype=int
        )

    # Step 3: Hybrid adaptive sampling
    elif method == "adaptive":

        # Approximately 50% uniform frames
        uniform_count = (n + 1) // 2

        uniform_indices = np.linspace(
            0, T - 1, uniform_count, dtype=int
        )

        # Calculate motion using hand landmarks only
        motion_scores = calculate_motion_scores(
            pose, valid_mask
        )

        # Exclude already selected uniform frames
        motion_scores[uniform_indices] = -np.inf

        # Remaining frames selected by highest motion
        motion_count = n - uniform_count

        motion_indices = np.argsort(
            -motion_scores,
            kind="stable"
        )[:motion_count]

        # Combine and sort chronologically
        indices = np.sort(
            np.concatenate([
                uniform_indices,
                motion_indices
            ])
        )

    # Step 4: Select ALL landmarks, including face
    sampled_pose = pose[indices]
    sampled_mask = valid_mask[indices]

    # Step 5: Pad shorter videos
    if n < target_frames:

        pad_count = target_frames - n

        sampled_pose = np.pad(
            sampled_pose,
            ((0, pad_count), (0, 0), (0, 0)),
            mode="constant"
        )

        sampled_mask = np.pad(
            sampled_mask,
            ((0, pad_count), (0, 0)),
            mode="constant",
            constant_values=False
        )

        # -1 represents a padded frame
        indices = np.pad(
            indices,
            (0, pad_count),
            mode="constant",
            constant_values=-1
        )

    return sampled_pose, sampled_mask, indices
