#!/usr/bin/env python3
"""Create fixed, shareable train/val/test manifests from selected_8000.csv.

Default: keep the GloSL dataset's provided train/dev/test assignments.
Alternative (--strategy grouped): create a deterministic video-group-disjoint
~80/10/10 split, using the same manifest and seed across team members.

No pose data is downloaded or modified.
"""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val", "dev": "val", "development": "val",
    "test": "test", "testing": "test",
}
SPLITS = ("train", "val", "test")


def load_manifest(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("The CSV has no header")
        rows = list(reader)
        columns = list(reader.fieldnames)
    if not rows:
        raise ValueError("Manifest is empty")
    required = {"pose_file", "text", "video_ref"}
    if not required.issubset(columns):
        raise ValueError(f"Missing columns: {sorted(required - set(columns))}")
    pose_paths = [row["pose_file"].strip() for row in rows]
    if any(not p for p in pose_paths) or len(set(pose_paths)) != len(rows):
        raise ValueError("Empty or repeated pose_file values in the manifest")
    if any(not row["text"].strip() for row in rows):
        raise ValueError("Some rows have no English translation")
    return rows, columns


def set_original_splits(rows, columns):
    if "split" not in columns:
        raise ValueError("The source manifest has no 'split' column")
    unknown = Counter()
    for row in rows:
        raw = row["split"].strip().lower()
        if raw not in SPLIT_ALIASES:
            unknown[raw or "<empty>"] += 1
        else:
            row["split"] = SPLIT_ALIASES[raw]
    if unknown:
        raise ValueError(f"Unknown split labels: {dict(unknown)}. Inspect the CSV before splitting.")


def set_grouped_splits(rows, columns, seed):
    # Groups by the video source so related clips never cross splits.
    grouped = defaultdict(list)
    for row in rows:
        ref = row["video_ref"].strip()
        if not ref:
            raise ValueError("Missing video_ref; cannot safely generate source-video-disjoint splits")
        grouped[ref].append(row)
    keys = sorted(grouped)
    random.Random(seed).shuffle(keys)
    goals = {"train": int(len(rows) * 0.8), "val": int(len(rows) * 0.1)}
    counts = Counter()
    stage = 0
    for key in keys:
        current = SPLITS[stage]
        for row in grouped[key]:
            row["split"] = current
        counts[current] += len(grouped[key])
        if stage == 0 and counts["train"] >= goals["train"]:
            stage = 1
        elif stage == 1 and counts["val"] >= goals["val"]:
            stage = 2
    if "split" not in columns:
        columns.append("split")


def verify(rows):
    sizes = Counter(row["split"] for row in rows)
    if any(sizes[s] == 0 for s in SPLITS):
        raise ValueError(f"Empty split found: {dict(sizes)}")
    video_memberships = defaultdict(set)
    missing_video_ref = 0
    for row in rows:
        ref = row["video_ref"].strip()
        if ref:
            video_memberships[ref].add(row["split"])
        else:
            missing_video_ref += 1
    overlaps = [(name, sorted(labels)) for name, labels in video_memberships.items() if len(labels) > 1]
    if overlaps:
        example = overlaps[:3]
        raise ValueError(f"Source-video leakage detected: {len(overlaps)} video_ref values appear in multiple splits. Examples: {example}. Use --strategy grouped for a custom group-safe split.")
    return sizes, len(video_memberships), missing_video_ref


def save_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Create matching train/val/test CSVs from the team's shared 8k sample manifest")
    parser.add_argument("--manifest", type=Path, default=Path("glosl_8000/selected_8000.csv"))
    parser.add_argument("--output", type=Path, default=Path("glosl_8000/splits"))
    parser.add_argument("--strategy", choices=("original", "grouped"), default="original")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows, columns = load_manifest(args.manifest)
    if len(rows) != 8000:
        raise ValueError(f"Expected 8000 selected examples, but found {len(rows)}")

    if args.strategy == "original":
        set_original_splits(rows, columns)
    else:
        set_grouped_splits(rows, columns, args.seed)
    rows.sort(key=lambda row: row["pose_file"])
    counts, video_groups, missing_ref = verify(rows)

    args.output.mkdir(parents=True, exist_ok=True)
    files = {"all": args.output / "shared_splits.csv"}
    save_csv(files["all"], rows, columns)
    for label in SPLITS:
        path = args.output / f"{label}.csv"
        save_csv(path, [row for row in rows if row["split"] == label], columns)
        files[label] = path

    summary = {
        "source_manifest_sha256": digest(args.manifest),
        "strategy": args.strategy,
        "seed_if_grouped": args.seed if args.strategy == "grouped" else None,
        "total": len(rows),
        "counts": {s: counts[s] for s in SPLITS},
        "distinct_known_video_refs": video_groups,
        "rows_missing_video_ref": missing_ref,
        "files_sha256": {label: digest(path) for label, path in files.items()},
    }
    (args.output / "split_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Total: {len(rows)} | train: {counts['train']} | val: {counts['val']} | test: {counts['test']}")
    print(f"Source-video overlap across splits: 0 (among {video_groups} identified videos)")
    if missing_ref:
        print(f"WARNING: {missing_ref} rows have no video_ref, so their source-video independence cannot be checked")
    print(f"Shared split SHA256: {summary['files_sha256']['all']}")
    print(f"Outputs in: {args.output}")
    print("Share shared_splits.csv and split_report.json with the team.")
    print("Do not regenerate different splits independently.")


if __name__ == "__main__":
    main()
