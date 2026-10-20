
import numpy as np
import sys

# Import Shahjad's pose extraction function
from src.features.pose_motion import extract_pose_from_video

# Import Rohit's uniform sampling function
from src.sampling.temporal import sample


def main():

    # Step 1: Get video path from command line
    if len(sys.argv) < 2:
        raise ValueError("Please provide a video path")

    video_path = sys.argv[1]

    # Step 2: Extract pose data using Shahjad's function
    # pose shape: (T, 75, 3)
    # mask shape: (T, 75)
    pose, valid_mask = extract_pose_from_video(video_path)
    # Simulate a short video using the first 20 real frames
    print("Original pose shape:", pose.shape)
    print("Original mask shape:", valid_mask.shape)

    # Step 3: Apply Rohit's uniform sampling
    sampled_pose, sampled_mask, indices = sample(
        pose,
        valid_mask,
        target_frames=64,
        method="uniform"
    )

    print("\nAfter uniform sampling:")
    print("Sampled pose shape:", sampled_pose.shape)
    print("Sampled mask shape:", sampled_mask.shape)
    print("Selected indices:", indices)

    # Step 4: Verify output dimensions
    assert sampled_pose.shape == (64, 75, 3)
    assert sampled_mask.shape == (64, 75)
    assert indices.shape == (64,)

    # Step 5: Calculate number of real frames
    # -1 represents padding, not an original frame
    actual_length = int(np.sum(indices >= 0))

    print("\nActual sequence length:", actual_length)
    print("Number of padded frames:", 64 - actual_length)

    # Step 6: Verify original frame ordering
    real_indices = indices[:actual_length]

    assert np.all(np.diff(real_indices) > 0)

    # Step 7: Verify correct pose-mask alignment
    assert np.array_equal(
        sampled_pose[:actual_length],
        pose[real_indices]
    )

    assert np.array_equal(
        sampled_mask[:actual_length],
        valid_mask[real_indices]
    )

    # Step 8: Verify zero-padding for short videos
    assert np.all(indices[actual_length:] == -1)
    assert np.all(sampled_pose[actual_length:] == 0)
    assert np.all(sampled_mask[actual_length:] == False)

    # Step 9: Verify correct sequence length
    assert actual_length == min(len(pose), 64)

    # Step 10: Display final result
    print("\nAll integration checks passed!")
    print("Final pose shape:", sampled_pose.shape)
    print("Length for Murad's GRU:", actual_length)


# Execute only when this file is run directly
if __name__ == "__main__":
    main()
