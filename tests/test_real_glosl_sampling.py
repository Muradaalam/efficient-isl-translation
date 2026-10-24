import numpy as np
from pathlib import Path

from src.features.glosl_pose import read_glosl_pose
from src.sampling.temporal import sample


def test_real_glosl_sampling():

    # Select one real GloSL pose file
    files = list(Path("glosl_8000/poses").rglob("*.pose"))
    assert files, "No GloSL pose files found"

    # Shahjad's preprocessing function
    pose, mask, confidence = read_glosl_pose(files[0])

    print("Original pose:", pose.shape)
    print("Original mask:", mask.shape)

    # Compare uniform and adaptive sampling
    for method in ["uniform", "adaptive"]:

        sampled_pose, sampled_mask, indices = sample(
            pose,
            mask,
            target_frames=64,
            method=method
        )

        valid_indices = indices[indices >= 0]

        print(f"\nMethod: {method}")
        print("Sampled pose:", sampled_pose.shape)
        print("Selected indices:", indices)
        print("Unique frames:", len(np.unique(valid_indices)))

        assert sampled_pose.shape == (64, 75, 3)
        assert sampled_mask.shape == (64, 75)
        assert np.all(np.diff(valid_indices) > 0)

        # Verify that frames and masks were copied correctly
        np.testing.assert_array_equal(
            sampled_pose[:len(valid_indices)],
            pose[valid_indices]
        )

        np.testing.assert_array_equal(
            sampled_mask[:len(valid_indices)],
            mask[valid_indices]
        )