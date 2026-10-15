
import numpy as np  # NumPy helps us work with arrays


def sample(pose, valid_mask, target_frames=64, method="uniform"):
    """
    Select frames from a pose sequence using uniform sampling.

    Inputs:
        pose: Pose coordinates, shape (T, 75, 3)
        valid_mask: Landmark validity, shape (T, 75)
        target_frames: Maximum number of frames to select
        method: Sampling method (currently only "uniform")

    Returns:
        sampled_pose: Selected pose frames
        sampled_mask: Masks corresponding to selected frames
        indices: Original frame positions that were selected
    """

    # Step 1: Check whether the requested method is supported
    if method != "uniform":
        raise ValueError("Only uniform sampling is supported")

    # Step 2: Convert inputs into NumPy arrays
    pose = np.asarray(pose)
    valid_mask = np.asarray(valid_mask)

    # Step 3: Check pose dimensions
    # T = number of frames
    # 75 = landmarks per frame
    # 3 = x, y, z coordinates per landmark
    if pose.ndim != 3 or pose.shape[1:] != (75, 3):
        raise ValueError("Expected pose shape (T, 75, 3)")

    # Step 4: Check that each pose landmark has a matching mask
    if valid_mask.shape != pose.shape[:2]:
        raise ValueError("Expected mask shape (T, 75)")

    # Step 5: Ensure the requested frame count is positive
    if not isinstance(target_frames, (int, np.integer)) or target_frames <= 0:
        raise ValueError("target_frames must be a positive integer")

    # Step 6: Count the original video frames
    T = pose.shape[0]

    # An empty video cannot be sampled
    if T == 0:
        raise ValueError("Cannot sample an empty sequence")

    # Step 7: Generate uniformly spaced frame indices
    # Example:
    # T = 10, target_frames = 5
    # indices = [0, 2, 4, 6, 9]
    # min() avoids repeating frames when T < target_frames
    indices = np.linspace(
        0,                      # First frame index
        T - 1,                  # Last frame index
        min(T, target_frames),  # Number of frames to select
        dtype=int               # Frame indices must be integers
    )

    # Step 8: Select complete pose frames at these indices
    sampled_pose = pose[indices]

    # Step 9: Select corresponding validity masks
    # Both arrays must use the same frame indices
    sampled_mask = valid_mask[indices]

    # Step 10: Return sampled data and original frame indices
    return sampled_pose, sampled_mask, indices
