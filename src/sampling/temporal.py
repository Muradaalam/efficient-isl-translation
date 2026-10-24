import numpy as np


def sample(pose, valid_mask, target_frames=64, method="uniform"):
    """
    Sample pose frames using uniform or hybrid adaptive sampling.

    Input:
        pose: (T, 75, 3)
        valid_mask: (T, 75)
        target_frames: Number of output frames
        method: "uniform" or "adaptive"

    Output:
        sampled_pose, sampled_mask, indices
    """

    pose = np.asarray(pose)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    # Step 1: Validate inputs
    if pose.ndim != 3 or pose.shape[1:] != (75, 3):
        raise ValueError("Expected pose shape (T, 75, 3)")

    if valid_mask.shape != pose.shape[:2]:
        raise ValueError("Expected valid_mask shape (T, 75)")

    if not isinstance(target_frames, (int, np.integer)) or target_frames <= 0:
        raise ValueError("target_frames must be positive")

    T = pose.shape[0]

    if T == 0:
        raise ValueError("Cannot sample an empty video")

    n = min(T, target_frames)

    # Step 2: Select frame indices
    if method == "uniform":

        indices = np.linspace(
            0, T - 1, n, dtype=int
        )

    elif method == "adaptive":

        # 50% frames for uniform temporal coverage
        uniform_count = (n + 1) // 2

        uniform_indices = np.linspace(
            0, T - 1, uniform_count, dtype=int
        )

        # Motion scores for all original frames
        motion_scores = calculate_motion_scores(
            pose, valid_mask
        )

        # Exclude already selected uniform frames
        motion_scores[uniform_indices] = -np.inf

        # Select remaining frames based on highest motion
        motion_count = n - uniform_count

        motion_indices = np.argsort(
            -motion_scores, kind="stable"
        )[:motion_count]

        # Combine and preserve chronological order
        indices = np.sort(
            np.concatenate([
                uniform_indices,
                motion_indices
            ])
        )

    else:
        raise ValueError("Method must be 'uniform' or 'adaptive'")

    # Step 3: Extract selected frames and masks
    sampled_pose = pose[indices]
    sampled_mask = valid_mask[indices]

    # Step 4: Pad sequences shorter than target_frames
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

        indices = np.pad(
            indices,
            (0, pad_count),
            mode="constant",
            constant_values=-1
        )

    return sampled_pose, sampled_mask, indices
 
def calculate_motion_scores(pose, valid_mask):
    """
    Calculate the movement score for each frame.
    Higher score = more landmark movement.
    """

    # Number of frames in the pose sequence
    T = pose.shape[0]

    # Initially, every frame has zero motion
    scores = np.zeros(T)

    # Calculate coordinate changes between consecutive frames
    differences = pose[1:] - pose[:-1]

    # Calculate Euclidean distance for each landmark
    distances = np.linalg.norm(differences, axis=2)

    # Only compare landmarks detected in both frames
    valid_pairs = valid_mask[1:] & valid_mask[:-1]

    # Ignore movement of missing landmarks
    distances = distances * valid_pairs

    # Average movement over valid landmarks
    counts = np.maximum(valid_pairs.sum(axis=1), 1)
    scores[1:] = distances.sum(axis=1) / counts

    return scores
