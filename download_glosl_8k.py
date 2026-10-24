#!/usr/bin/env python3
"""Download the exact same GloSL-ISL pose/English pairs across teammates.

First member:
    python3 download_glosl_8k.py --prepare-only
    # Share glosl_8000/selected_8000.csv alongside this script.
    python3 download_glosl_8k.py

Other teammates:
    python3 download_glosl_8k.py --manifest selected_8000.csv

The repository commit and ordered pose IDs are saved in the manifest. Downloads
resume by skipping files already present. Missing files are never substituted.
"""

import argparse
import csv
import hashlib
import random
import shutil
import time
from pathlib import Path, PurePosixPath

REPO = "pedroodb/glosl-isl"


def pose_path(raw):
    path = str(raw or "").strip().removeprefix("./")
    parts = PurePosixPath(path).parts
    if len(parts) < 3 or parts[0] != "poses" or ".." in parts or not path.endswith(".pose"):
        raise ValueError(f"Invalid pose path: {raw!r}")
    return path


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader), list(reader.fieldnames)


def write_csv(path, rows, columns):
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def make_manifest(output, count, seed):
    from huggingface_hub import HfApi, hf_hub_download

    revision = HfApi().dataset_info(REPO).sha
    if not revision:
        raise RuntimeError("Could not obtain a fixed dataset revision.")

    print(f"Dataset commit: {revision}")
    annotation_path = hf_hub_download(repo_id=REPO, filename="annotations.csv",
                                      repo_type="dataset", revision=revision, local_dir=output)
    rows, columns = load_csv(annotation_path)
    candidates = []
    for row in rows:
        if not str(row.get("text", "")).strip():
            continue
        try:
            pose_path(row.get("pose_file"))
        except ValueError:
            continue
        candidates.append(row)

    print(f"Annotations: {len(rows):,}; pose/text pairs: {len(candidates):,}")
    if len(candidates) < count:
        raise RuntimeError(f"Only {len(candidates)} usable rows, need {count}.")

    # Matches the earlier probe_glosl_subset.py selection when the CSV is unchanged.
    random.Random(seed).shuffle(candidates)
    selected = [dict(row, repo_revision=revision) for row in candidates[:count]]
    manifest = output / "selected_8000.csv"
    write_csv(manifest, selected, columns + ["repo_revision"])
    print(f"Wrote locked {count:,}-record manifest: {manifest}")
    print(f"Manifest SHA256: {hashlib.sha256(manifest.read_bytes()).hexdigest()}")
    return manifest


def verify_manifest(manifest, count):
    rows, columns = load_csv(manifest)
    if len(rows) != count:
        raise ValueError(f"Manifest has {len(rows)} entries, expected {count}.")
    if not {"pose_file", "text", "repo_revision"}.issubset(columns):
        raise ValueError("Manifest needs pose_file, text, and repo_revision columns.")
    revisions = {row["repo_revision"].strip() for row in rows}
    if len(revisions) != 1 or not next(iter(revisions)):
        raise ValueError("Manifest has missing or inconsistent repository revisions.")
    paths = [pose_path(row["pose_file"]) for row in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("Manifest contains repeated pose paths; needs unique sample files.")
    if any(not row["text"].strip() for row in rows):
        raise ValueError("Manifest contains an empty English translation.")
    return rows, columns, next(iter(revisions))


def download_subset(rows, columns, revision, output, retries):
    from huggingface_hub import hf_hub_download

    completed, failed = [], []
    total_size = 0
    already = 0
    for index, row in enumerate(rows, 1):
        name = pose_path(row["pose_file"])
        destination = output / name
        if destination.is_file() and destination.stat().st_size > 0:
            size = destination.stat().st_size
            already += 1
        else:
            size = None
            for attempt in range(1, retries + 1):
                try:
                    downloaded = hf_hub_download(repo_id=REPO, filename=name,
                                                 repo_type="dataset", revision=revision,
                                                 local_dir=output)
                    size = Path(downloaded).stat().st_size
                    if size == 0:
                        raise IOError("Downloaded file is empty")
                    break
                except Exception as exc:
                    print(f"FAIL {index}/{len(rows)} ({attempt}/{retries}): {name}: {exc}")
                    if attempt < retries:
                        time.sleep(min(2 ** attempt, 15))

        if size is None:
            failed.append(name)
        else:
            total_size += size
            completed.append(dict(row, bytes=size))

        if index % 100 == 0 or index == len(rows):
            print(f"Processed {index:,}/{len(rows):,} | good: {len(completed):,} | "
                  f"failed: {len(failed)} | pose size: {total_size/1e9:.2f} GB", flush=True)

    write_csv(output / "downloaded_subset.csv", completed, columns + ["bytes"])
    (output / "failed_downloads.txt").write_text("\n".join(failed) + ("\n" if failed else ""), encoding="utf-8")
    print(f"\nAlready present: {already:,} files")
    print(f"Successfully present: {len(completed):,}/{len(rows):,}")
    print(f"Downloaded pose storage: {total_size/1e9:.2f} GB")
    print(f"Translation manifest: {output / 'downloaded_subset.csv'}")
    if failed:
        print(f"Some files failed; rerun the same command to retry them. See: {output / 'failed_downloads.txt'}")
        return 1
    print("SUCCESS: every selected pose file has its matching English annotation.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Reproducibly download 8,000 GloSL pose-English pairs.")
    parser.add_argument("--output", type=Path, default=Path("glosl_8000"))
    parser.add_argument("--manifest", type=Path, help="A teammate's selected_8000.csv (guarantees identical examples)")
    parser.add_argument("--prepare-only", action="store_true", help="Select and save the manifest; do not download poses")
    parser.add_argument("--count", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    if args.count < 1 or args.retries < 1:
        parser.error("--count and --retries must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = args.manifest or args.output / "selected_8000.csv"
    if not manifest.is_file():
        if args.manifest:
            parser.error(f"Manifest not found: {args.manifest}")
        manifest = make_manifest(args.output, args.count, args.seed)

    rows, columns, revision = verify_manifest(manifest, args.count)
    print(f"Using manifest: {manifest}")
    print(f"Manifest SHA256: {hashlib.sha256(manifest.read_bytes()).hexdigest()}")
    print(f"Pinned dataset revision: {revision}")

    if manifest.resolve() != (args.output / "selected_8000.csv").resolve():
        shutil.copyfile(manifest, args.output / "selected_8000.csv")

    if args.prepare_only:
        print("Share selected_8000.csv and this script with your teammates.")
        return 0

    free_gb = shutil.disk_usage(args.output).free / 1e9
    print(f"Free disk space: {free_gb:.1f} GB (8,000 poses may need ~13 GB or more)")
    return download_subset(rows, columns, revision, args.output, args.retries)


if __name__ == "__main__":
    raise SystemExit(main())
