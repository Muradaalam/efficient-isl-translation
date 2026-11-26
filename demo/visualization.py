import argparse
import subprocess
from pathlib import Path

from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer


def visualize_pose(pose_path, output_path, video_path=None):
    pose_path = Path(pose_path)
    output_path = Path(output_path)

    if not pose_path.is_file():
        raise FileNotFoundError(pose_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())

    visualizer = PoseVisualizer(pose)

    if video_path is None:
        frames = visualizer.draw(background_color=(0, 0, 0))
    else:
        frames = visualizer.draw_on_video(str(video_path))

    raw_path = output_path.with_name(output_path.stem + "_raw.mp4")

    visualizer.save_video(str(raw_path), frames)

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", str(raw_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path)
    ], check=True)

    raw_path.unlink()

    print("Visualization saved:", output_path)
    print("Total frames:", pose.body.data.shape[0])
    print("FPS:", pose.body.fps)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pose", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--video", default=None)

    args = parser.parse_args()

    visualize_pose(args.pose, args.output, args.video)