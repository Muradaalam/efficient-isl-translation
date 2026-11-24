
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.features.glosl_pose import read_glosl_pose
from src.sampling.temporal import sample, calculate_motion_scores

# Load one real GloSL pose file
pose_file = next(Path("glosl_8000/poses").rglob("*.pose"))
pose, mask, confidence = read_glosl_pose(pose_file)

print("Original frames:", len(pose))

# Calculate motion scores for all original frames
motion_scores = calculate_motion_scores(pose, mask)

# Uniform sampling
_, _, uniform_indices = sample(
    pose, mask, target_frames=64, method="uniform"
)

# Hybrid adaptive sampling
_, _, adaptive_indices = sample(
    pose, mask, target_frames=64, method="adaptive"
)

# Remove padding indices, if present
uniform_indices = uniform_indices[uniform_indices >= 0]
adaptive_indices = adaptive_indices[adaptive_indices >= 0]

# Plot motion scores and selected frames
plt.figure(figsize=(14, 5))

plt.plot(
    np.arange(len(motion_scores)),
    motion_scores,
    label="Motion score",
    linewidth=1.5
)

plt.scatter(
    uniform_indices,
    motion_scores[uniform_indices],
    label="Uniform selected frames",
    marker="o",
    s=24
)

plt.scatter(
    adaptive_indices,
    motion_scores[adaptive_indices],
    label="Adaptive selected frames",
    marker="x",
    s=35
)

plt.xlabel("Original frame index")
plt.ylabel("Average landmark displacement")
plt.title("Uniform vs Hybrid Adaptive Sampling — GloSL-ISL")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

# Save the figure for your report and viva
output_path = Path("experiments/sampling_comparison.png")
output_path.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output_path, dpi=200)
plt.show()

print("Figure saved to:", output_path)
