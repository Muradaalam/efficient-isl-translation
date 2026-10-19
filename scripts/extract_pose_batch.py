
import argparse
import csv
from pathlib import Path

import numpy as np

from src.features.pose_motion import extract_pose_from_video


def extract_batch(video_dir, output_dir, limit=None):
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)

    if not video_dir.is_dir():
        raise ValueError(f"Video directory not found: {video_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        path for path in video_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".mp4"
    )

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        videos = videos[:limit]

    records = []

    for index, video in enumerate(videos, start=1):
        relative_path = video.relative_to(video_dir)
        output_path = output_dir / relative_path.with_suffix(".npz")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{index}/{len(videos)}] {relative_path}", flush=True)

        try:
            pose, mask = extract_pose_from_video(video)

            np.savez_compressed(output_path, pose=pose, valid_mask=mask)

            records.append({
                "video_path": str(relative_path),
                "feature_path": str(output_path.relative_to(output_dir)),
                "frames": len(pose),
                "status": "success",
                "error": ""
            })

        except Exception as error:
            records.append({
                "video_path": str(relative_path),
                "feature_path": "",
                "frames": 0,
                "status": "failed",
                "error": str(error)
            })

            print(f"FAILED: {error}", flush=True)

    report_path = output_dir / "extraction_report.csv"

    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["video_path", "feature_path", "frames", "status", "error"]
        )
        writer.writeheader()
        writer.writerows(records)

    successful = sum(record["status"] == "success" for record in records)
    print(f"Completed: {successful}/{len(records)} videos")
    print(f"Report: {report_path}")

    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch ISL pose extraction")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    extract_batch(args.video_dir, args.output_dir, args.limit)
