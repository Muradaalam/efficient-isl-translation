
import numpy as np
import sys

# Import Shahjad's pose extraction function
from src.features.pose_motion import extract_pose_from_video

# Import your uniform sampling function
from src.sampling.temporal import sample

# Take the video path from the command line
video_path = sys.argv[1]

# Step 2: Extract poses and validity masks from the video
# Expected shapes: (T, 75, 3) and (T, 75)
pose, valid_mask = extract_pose_from_video(video_path)

# Step 3: Display the original data dimensions
print("Original pose shape:", pose.shape)
print("Original mask shape:", valid_mask.shape)

# Step 4: Apply your uniform sampling algorithm
sampled_pose, sampled_mask, indices = sample(
    pose,
    valid_mask,
    target_frames=64,
    method="uniform"
)

# Step 5: Display the sampled data dimensions
print("Sampled pose shape:", sampled_pose.shape)
print("Sampled mask shape:", sampled_mask.shape)
print("Selected frame indices:", indices)

# Step 6: Verify that pose and mask are aligned
assert sampled_pose.shape[0] == sampled_mask.shape[0]

# Step 7: Verify that no more than 64 frames remain
assert sampled_pose.shape[0] <= 64

print("Integration test passed!")
