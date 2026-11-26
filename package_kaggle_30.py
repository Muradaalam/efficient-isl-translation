"""Run from efficient-isl-translation repo root: python3 package_kaggle_30.py"""
import csv
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

MANIFEST = Path('test/pose_train_30.csv')
OUTPUT = Path('isl30_kaggle.zip')
CODE = [
    Path('experiments/flan_t5_training/dataset.py'),
    Path('experiments/flan_t5_training/model.py'),
    Path('experiments/flan_t5_training/train.py'),
    Path('experiments/flan_t5_training/evaluate.py'),
]

def main():
    missing = [str(path) for path in [MANIFEST, *CODE] if not path.is_file()]
    if missing: raise SystemExit('Missing required files: ' + ', '.join(missing))
    with MANIFEST.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 30: raise SystemExit(f'Expected exactly 30 manifest rows, found {len(rows)}')
    pose_paths = [Path(row['pose_file']) for row in rows]
    missing = [str(path) for path in pose_paths if not path.is_file()]
    if missing: raise SystemExit('Missing pose files: ' + ', '.join(missing))
    if any(not row['text'].strip() for row in rows): raise SystemExit('At least one label is empty')
    if len(set(map(str, pose_paths))) != 30: raise SystemExit('Duplicate pose paths in manifest')
    for path in pose_paths:
        if path.is_absolute() or '..' in path.parts: raise SystemExit(f'Unsafe relative pose path: {path}')
    with ZipFile(OUTPUT, 'w', compression=ZIP_DEFLATED) as z:
        for path in [MANIFEST, *CODE, *pose_paths]:
            z.write(path, arcname=str(path))
    print('Created:', OUTPUT.resolve())
    print('Rows:', len(rows))
    print('Distinct labels:', len(set(row['text'].strip() for row in rows)))
    print('Size (MB):', round(OUTPUT.stat().st_size / 1_000_000, 1))
    print('Upload this ZIP as a Kaggle dataset; add it to the notebook inputs.')

if __name__ == '__main__': main()
