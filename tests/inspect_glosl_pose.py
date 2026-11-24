from pathlib import Path
from pose_format import Pose

# Automatically find the first downloaded pose file
pose_path = next(Path("glosl_8000/poses").rglob("*.pose"))

print("Reading:", pose_path)

# Load the pose sequence
with open(pose_path, "rb") as f:
    pose = Pose.read(f.read())

# Get landmark coordinates and confidence values
data = pose.body.data

print("Pose shape:", data.shape)
print("FPS:", pose.body.fps)

if pose.body.confidence is not None:
    print("Confidence shape:", pose.body.confidence.shape)