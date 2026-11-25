"""Audit a fixed GloSL manifest and optionally extend its training split only.
Run from repository root. Requires pandas, huggingface_hub, numpy, tqdm,
and the existing scripts.train_flan_v4 module for pose validation.
"""
import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

REPO = 'pedroodb/glosl-isl'


def normalized(s):
    return ' '.join(str(s).split())


def clip_id(s):
    return Path(str(s)).stem


def source_group(s):
    # GloSL ISL IDs end with a numeric clip index; support both separators.
    m = re.fullmatch(r'(.+?)-+\d+', str(s))
    if not m:
        raise ValueError(f'Unrecognized clip ID: {s!r}')
    return m.group(1)


def wordset(s):
    return set(re.findall(r"[a-z]+(?:'[a-z]+)?", str(s).lower()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--pose-root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--extra', type=int, default=0)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--revision', default=None)
    p.add_argument('--audit-samples', type=int, default=100)
    a = p.parse_args()
    if a.extra < 0 or a.workers < 1 or a.audit_samples < 1:
        p.error('extra >= 0, workers >= 1, audit-samples >= 1')
    a.output.mkdir(parents=True, exist_ok=True)
    from scripts.train_flan_v4 import read_raw, normalize_pose, resolve_path, FACE25
    local = pd.read_csv(a.manifest, dtype={'id': str})
    required = {'pose_file', 'text', 'split'}
    if not required <= set(local.columns):
        raise ValueError(f'Missing manifest columns: {required - set(local.columns)}')
    if set(local.split) != {'train', 'val', 'test'}:
        raise ValueError(f'Unexpected split labels: {local.split.unique()}')
    if local[list(required)].isna().any().any():
        raise ValueError('Null manifest fields')
    ids = local.pose_file.map(clip_id)
    if 'id' in local and not local.id.eq(ids).all():
        raise ValueError('Manifest id does not match pose filename')
    if ids.duplicated().any():
        raise ValueError('Duplicate clip IDs in manifest')
    groups = ids.map(source_group)
    train = local.split.eq('train')
    holdout = ~train
    shared = set(groups[train]) & set(groups[holdout])
    print('Original split counts:', local.split.value_counts().to_dict(), flush=True)
    print('Train/holdout source-ID overlap:', len(shared), flush=True)
    vocab = set().union(*(wordset(t) for t in local.loc[train, 'text']))
    for split in ['train', 'val']:
        rows = local[local.split.eq(split)]
        tokens = Counter(w for t in rows.text for w in re.findall(r"[a-z]+(?:'[a-z]+)?", str(t).lower()))
        print(split, 'unique words:', len(tokens), 'OOV token rate:', round(sum(n for w,n in tokens.items() if w not in vocab)/max(sum(tokens.values()),1),4), flush=True)
    stats = []
    ix = np.r_[np.arange(75), 75+np.asarray(FACE25)]
    for split in ['train','val']:
        rows = local[local.split.eq(split)]
        rows = rows.sample(min(len(rows), a.audit_samples), random_state=a.seed)
        for row in tqdm(rows.itertuples(), total=len(rows), desc=f'Audit {split}'):
            try:
                path = resolve_path(a.pose_root, row.pose_file)
                q,v = read_raw(path)
                n = normalize_pose(q,v)
                stats.append(dict(split=split, id=clip_id(path), frames=len(q),
                    valid_frames=int(v[:,ix].any(1).sum()), hand_frame_fraction=float(v[:,33:75].any(1).mean()),
                    hand_landmark_fraction=float(v[:,33:75].mean()),
                    normalized_xy_p99=float(np.percentile(np.abs(n[v,:2]),99)),
                    normalized_z_p99=float(np.percentile(np.abs(n[v,2]),99)), error=''))
            except Exception as e:
                stats.append(dict(split=split,id=clip_id(row.pose_file),error=str(e)))
    stats = pd.DataFrame(stats)
    stats.to_csv(a.output/'pose_audit.csv', index=False)
    print(stats.groupby('split').median(numeric_only=True).to_string(), flush=True)
    print('Pose audit errors:', int(stats.error.ne('').sum()), flush=True)
    if a.extra == 0:
        print('Audit complete. Original manifest unchanged.')
        return
    if shared or stats.error.ne('').any():
        raise ValueError('Resolve source overlap / pose errors before expanding')
    revision = a.revision or HfApi().dataset_info(REPO).sha
    metadata = hf_hub_download(REPO, 'annotations.csv', repo_type='dataset', revision=revision)
    remote = pd.read_csv(metadata, dtype={'id': str})
    if not {'id','split','text','pose_file','video_ref'} <= set(remote.columns):
        raise ValueError('Remote metadata schema changed')
    if remote.id.isna().any() or remote.id.duplicated().any():
        raise ValueError('Invalid remote IDs')
    remote = remote.set_index('id', drop=False)
    if not ids.isin(remote.index).all():
        raise ValueError('Current dataset release differs: some original IDs not found; stop')
    original = remote.loc[ids.tolist()]
    if not np.array_equal(original.text.map(normalized).to_numpy(), local.text.map(normalized).to_numpy()):
        raise ValueError('Local/remote annotation text mismatch; stop')
    print('All original IDs and text match remote revision:', revision, flush=True)
    refs = original.video_ref.fillna('').astype(str).str.strip().reset_index(drop=True)
    if refs[holdout].eq('').any():
        raise ValueError('Missing source-video reference for holdout clip; cannot verify disjointness')
    original_holdout_refs = set(refs[holdout]) - {''}
    original_train_refs = set(refs[train]) - {''}
    if original_holdout_refs & original_train_refs:
        raise ValueError('Existing source-video reference overlap; stop')
    candidate = remote[remote.split.eq('train') & ~remote.id.isin(ids)].copy()
    candidate = candidate.dropna(subset=['text','pose_file','video_ref'])
    candidate = candidate[candidate.text.map(normalized).ne('') & candidate.video_ref.astype(str).str.strip().ne('')]
    candidate = candidate[~candidate.id.map(source_group).isin(set(groups[holdout]))]
    candidate = candidate[~candidate.video_ref.astype(str).str.strip().isin(original_holdout_refs)]
    candidate = candidate.sample(frac=1, random_state=a.seed).reset_index(drop=True)
    if len(candidate) < a.extra:
        raise ValueError(f'Only {len(candidate)} eligible candidates')
    print('Eligible additional train clips:', len(candidate), flush=True)
    # Local cache avoids changing original dataset; all new paths are absolute.
    pose_dir = a.output/'downloaded'
    pose_dir.mkdir(exist_ok=True)
    def get(row):
        try:
            path = hf_hub_download(REPO, str(row.pose_file), repo_type='dataset', revision=revision, local_dir=pose_dir)
            coords, valid = read_raw(path)
            if not valid[:,ix].any() or not valid[:,33:75].any():
                raise ValueError('No selected/hand landmarks')
            normalized_pose = normalize_pose(coords,valid)
            if not np.isfinite(normalized_pose).all():
                raise ValueError('Nonfinite normalized pose')
            record = row._asdict()
            record.pop('Index',None)
            record['pose_file'] = str(Path(path).resolve())
            record['split'] = 'train'
            return record, None
        except Exception as e:
            return None, dict(id=row.id,error=str(e))
    accepted, rejected = [], []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        with tqdm(total=a.extra, desc='Validated extra training clips') as bar:
            i = 0
            while i < len(candidate) and len(accepted) < a.extra:
                batch = candidate.iloc[i:i+min(a.workers*4, a.extra-len(accepted))]
                i += len(batch)
                for record,error in pool.map(get,batch.itertuples(index=False)):
                    if error: rejected.append(error)
                    elif len(accepted)<a.extra:
                        accepted.append(record); bar.update(1)
                if len(rejected) >= max(100, a.extra) and len(rejected) > 3*len(accepted):
                    print('Too many failed downloads/invalid poses; stopping for inspection', flush=True)
                    break
    pd.DataFrame(rejected, columns=['id','error']).to_csv(a.output/'rejected.csv', index=False)
    if len(accepted) != a.extra:
        raise RuntimeError(f'Only {len(accepted)} / {a.extra} extra clips validated')
    extra = pd.DataFrame(accepted)
    # Preserve original columns and values. Extra rows can introduce metadata columns.
    combined = pd.concat([local, extra], ignore_index=True, sort=False)
    assert combined.loc[combined.split.ne('train'), local.columns].reset_index(drop=True).equals(local.loc[holdout].reset_index(drop=True)) or combined.loc[combined.split.ne('train'), local.columns].reset_index(drop=True).astype(str).equals(local.loc[holdout].reset_index(drop=True).astype(str))
    assert not combined.pose_file.map(clip_id).duplicated().any()
    dest = a.output/'expanded_manifest.csv'
    combined.to_csv(dest,index=False)
    (a.output/'expansion_info.json').write_text(json.dumps(dict(repo=REPO,revision=revision,seed=a.seed,added=len(extra),counts=combined.split.value_counts().to_dict()),indent=2))
    print('Expanded manifest:', dest, flush=True)
    print(combined.split.value_counts().to_string(), flush=True)


if __name__ == '__main__':
    main()
