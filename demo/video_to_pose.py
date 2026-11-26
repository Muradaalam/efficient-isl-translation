import argparse
import subprocess
from pathlib import Path

from pose_format import Pose


def convert_video_to_pose(video_path, output_path):
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run([
        "video_to_pose",
        "--format", "mediapipe",
        "-i", str(video_path),
        "-o", str(output_path)
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Pose extraction failed:\n{result.stdout}\n{result.stderr}"
        )

    with open(output_path, "rb") as f:
        pose = Pose.read(f.read())

    print("Pose saved:", output_path)
    print("Pose shape:", pose.body.data.shape)
    print("FPS:", pose.body.fps)

    for component in pose.header.components:
        print(component.name, len(component.points))

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    convert_video_to_pose(args.video, args.output)