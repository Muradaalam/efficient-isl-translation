
import numpy as np
import pytest

# Import your sampling function
from src.sampling.temporal import sample


def test_uniform_sampling():

    # Step 1: Create a dummy pose array like Shahjad's output
    # 120 frames, 75 landmarks, 3 coordinates (x, y, z)
    pose = np.random.rand(120, 75, 3)

    # Step 2: Create a validity mask
    # True means a landmark was detected successfully
    valid_mask = np.ones((120, 75), dtype=bool)

    # Step 3: Call your uniform sampling function
    # We want to select 64 frames from 120
    sampled_pose, sampled_mask, indices = sample(
        pose,
        valid_mask,
        target_frames=64,
        method="uniform"
    )

    # Step 4: Verify the sampled pose dimensions
    assert sampled_pose.shape == (64, 75, 3)

    # Step 5: Verify the sampled mask dimensions
    assert sampled_mask.shape == (64, 75)

    # Step 6: Verify that 64 frame indices were selected
    assert len(indices) == 64

    # Step 7: Verify chronological frame ordering
    assert np.all(np.diff(indices) > 0)

    # Step 8: Verify first and last frames are included
    assert indices[0] == 0
    assert indices[-1] == 119

    # Step 9: Verify the selected frames are correct
    assert np.array_equal(sampled_pose, pose[indices])

    # Step 10: Verify the masks match the selected frames
    assert np.array_equal(sampled_mask, valid_mask[indices])



def test_short_video():

    # Create dummy pose data with only 20 frames
    pose = np.random.rand(20, 75, 3)
    mask = np.ones((20, 75), dtype=bool)

    # Request 64 frames from a 20-frame video
    sampled_pose, sampled_mask, indices = sample(
        pose, mask, target_frames=64
    )

    # Output must always contain 64 frames
    assert sampled_pose.shape == (64, 75, 3)
    assert sampled_mask.shape == (64, 75)
    assert indices.shape == (64,)

    # First 20 frames must preserve original data
    assert np.array_equal(sampled_pose[:20], pose)
    assert np.array_equal(sampled_mask[:20], mask)

    # Remaining 44 frames must be zero padding
    assert np.all(sampled_pose[20:] == 0)

    # Padding masks must be False
    assert np.all(sampled_mask[20:] == False)

    # First 20 indices must match original frames
    assert np.array_equal(indices[:20], np.arange(20))

    # Padded frames must have index -1
    assert np.all(indices[20:] == -1)

    # Actual sequence length for Murad's GRU
    actual_length = np.sum(indices >= 0)
    assert actual_length == 20

def test_empty_video():

    # Create an empty pose array (0 frames)
    pose = np.empty((0, 75, 3))

    # Create an empty validity mask
    mask = np.empty((0, 75), dtype=bool)

    # Our sampler should reject empty sequences
    with pytest.raises(ValueError):
        sample(pose, mask, target_frames=64)
