#!/usr/bin/env python3
"""Collect 100 real Indian Sign Language .mp4 clips in test/.

Source: vidit031/isl-isolated-40words on Hugging Face.
Only the INCLUDE-origin subset is selected to avoid mixing dataset sources.
These are isolated-sign clips, not continuous signed sentences.
"""
import argparse
import csv
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path, PurePosixPath

REPO = 'vidit031/isl-isolated-40words'
DATASET_URL = 'https://huggingface.co/datasets/' + REPO


def select_clips(files, count):
    groups = defaultdict(list)
    for filename in sorted(files):
        path = PurePosixPath(filename)
        if path.suffix.lower() != '.mp4' or '__INCLUDE__' not in path.name.upper():
            continue
        label = path.parts[0] if len(path.parts) > 1 else 'unknown'
        groups[label].append(filename)

    # Interleave classes, so we don't take 100 clips from just a few words.
    ordered = []
    i = 0
    while len(ordered) < count and any(i < len(v) for v in groups.values()):
        for label in sorted(groups):
            if i < len(groups[label]):
                ordered.append((label, groups[label][i]))
                if len(ordered) == count: break
        i += 1
    return ordered


def valid_mp4(path):
    try:
        if path.stat().st_size < 512: return False
        with path.open('rb') as f: header = f.read(16)
        return header[4:8] == b'ftyp'
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--output', default='test', help='Folder to save MP4 files')
    args = parser.parse_args()
    if args.count < 1: parser.error('--count must be positive')

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        sys.exit('Missing dependency. Run: python3 -m pip install huggingface_hub')

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print(f'Listing ISL videos in {REPO} ...', flush=True)
    try:
        filenames = list_repo_files(REPO, repo_type='dataset')
    except Exception as exc:
        sys.exit(f'Could not list the dataset files: {exc}')

    selected = select_clips(filenames, args.count)
    if len(selected) != args.count:
        sys.exit(f'The INCLUDE subset contains only {len(selected)} available clips; need {args.count}.')

    manifest_rows = []
    failed = []
    for index, (label, filename) in enumerate(selected, start=1):
        name = f'sample_{index:03d}.mp4'
        target = output / name
        if valid_mp4(target):
            print(f'[{index}/{args.count}] already exists: {name}', flush=True)
        else:
            if target.exists(): target.unlink()
            for attempt in range(1, 4):
                try:
                    local_file = hf_hub_download(repo_id=REPO, repo_type='dataset', filename=filename)
                    temp = output / f'.{name}.tmp'
                    shutil.copy2(local_file, temp)
                    if not valid_mp4(temp):
                        temp.unlink(missing_ok=True)
                        raise ValueError('Downloaded content does not appear to be MP4')
                    os.replace(temp, target)
                    print(f'[{index}/{args.count}] {name}: {label}', flush=True)
                    break
                except Exception as exc:
                    print(f'  Attempt {attempt}/3 failed for {filename}: {exc}', file=sys.stderr)
                    if attempt == 3:
                        failed.append(filename)
                    else:
                        time.sleep(attempt * 2)

        if valid_mp4(target):
            manifest_rows.append({
                'filename': name,
                'label': label,
                'source_file': filename,
                'source_dataset': REPO,
                'source_url': f'{DATASET_URL}/blob/main/{filename}',
                'upstream': 'INCLUDE',
                'provenance_note': 'Isolated Indian Sign Language word clip, not sentence-level translation',
            })

    manifest = output / 'manifest.csv'
    fields = ['filename', 'label', 'source_file', 'source_dataset', 'source_url', 'upstream', 'provenance_note']
    with manifest.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f'\nValid MP4 files collected: {len(manifest_rows)}/{args.count}', flush=True)
    print(f'Manifest: {manifest}', flush=True)
    if failed or len(manifest_rows) != args.count:
        sys.exit('Some files could not be downloaded. Rerun the same command to resume.')
    print('Completed successfully.', flush=True)


if __name__ == '__main__':
    main()
