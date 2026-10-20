
import argparse
import csv
from pathlib import Path

import numpy as np

from src.features.pose_motion import extract_pose_from_video


def extract_manifest(manifest_path, video_dir, output_dir, limit=None):
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    records = [row for row in rows if row["status"] == "valid"]

    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        records = records[:limit]

    report = []

    for index, row in enumerate(records, start=1):
        relative_path = Path(row["video_path"]).relative_to("Videos_Sentence_Level")
        video_path = video_dir / relative_path
        feature_path = output_dir / relative_path.with_suffix(".npz")
        feature_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{index}/{len(records)}] {relative_path}", flush=True)

        try:
            if feature_path.is_file():
                with np.load(feature_path) as data:
                    frames = len(data["pose"])
                status = "existing"
            else:
                pose, mask = extract_pose_from_video(video_path)
                np.savez_compressed(feature_path, pose=pose, valid_mask=mask)
                frames = len(pose)
                status = "success"

            error = ""

        except Exception as exc:
            frames = 0
            status = "failed"
            error = str(exc)
            print(f"FAILED: {error}", flush=True)

        report.append({
            "sample_id": row["sample_id"],
            "split": row["split"],
            "video_path": row["video_path"],
            "frames": frames,
            "status": status,
            "error": error
        })

    report_path = output_dir / "manifest_extraction_report.csv"

    with open(report_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)

    successful = sum(row["status"] in ("success", "existing") for row in report)

    print(f"Available features: {successful}/{len(report)}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="experiments/isl_manifest_split.csv")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-dir", default="data/pose_features")
    parser.add_argument("--limit", type=int)

    args = parser.parse_args()
    extract_manifest(args.manifest, args.video_dir, args.output_dir, args.limit)
