"""Extract SpaMo-OF features with coverage + motion adaptive selection."""
import argparse
import csv
import importlib.util
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select(scores, budget, method):
    n = len(scores)
    if budget <= 0: raise ValueError('Budget must be positive')
    if method == 'all' or n <= budget: return np.arange(n)
    if method == 'uniform' or np.allclose(scores, scores[0]):
        return np.linspace(0, n - 1, budget).round().astype(int)
    if method == 'motion-aware':
        return np.sort(np.argsort(-scores, kind='stable')[:budget])
    uniform = np.linspace(0, n - 1, max(1, budget // 2)).round().astype(int)
    remaining = np.setdiff1d(np.arange(n), uniform)
    high = remaining[np.argsort(-scores[remaining], kind='stable')[:budget-len(uniform)]]
    return np.sort(np.concatenate([uniform, high]))


def windows(n):
    # Sixteen consecutive original flow maps, stride eight; repeat last for tail.
    starts = range(0, max(1, n-16+1), 8)
    starts = list(starts)
    if starts[-1] + 16 < n: starts.append(starts[-1]+8)
    return [np.minimum(np.arange(s,s+16), n-1) for s in starts]


def key(row):
    return hashlib.sha256(row['id'].encode()).hexdigest()[:24]


def read_manifest(root):
    with (Path(root)/'manifest.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len({r['id'] for r in rows}) != len(rows): raise ValueError('Duplicate video IDs')
    return rows


def read_video(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames, scores, last = [], [], None
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            # Lightweight sampling proxy; not the GMA motion encoder.
            grey = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64,64)).astype(np.float32)
            scores.append(0. if last is None else float(np.mean(np.abs(grey-last))))
            last = grey
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames: raise ValueError(f'Cannot decode {path}')
    return frames, np.asarray(scores, dtype=np.float32)


def extractor(a):
    import torch
    import cv2
    from transformers import AutoImageProcessor, CLIPVisionModel, VideoMAEModel, VideoMAEImageProcessor
    from tqdm import tqdm
    from argparse import Namespace

    # GMA imports its own top-level utils; load S2 directly to avoid a collision.
    sys.path.insert(0, str(Path(a.gma).resolve()/'core'))
    from network import RAFTGMA
    from utils import flow_viz
    s2 = load_module('upstream_s2', Path(a.spamo)/'utils/s2wrapper.py')
    cv2.setNumThreads(1)
    device = torch.device('cuda:0')
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    gma = RAFTGMA(Namespace(num_heads=1, position_only=False, position_and_content=False, mixed_precision=(a.precision == "fp16" and not a.gma_fp32), dropout=0))
    state = torch.load(a.gma_checkpoint, map_location='cpu', weights_only=True)
    gma.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()})
    gma = gma.to(device).eval()
    # Keep master weights FP32; use autocast only during frozen inference.
    fast = a.precision == "fp16"
    clip_name = 'openai/clip-vit-large-patch14'
    mae_name = 'MCG-NJU/videomae-large'
    clip = CLIPVisionModel.from_pretrained(clip_name).to(device).eval()
    cp = AutoImageProcessor.from_pretrained(clip_name)
    mae = VideoMAEModel.from_pretrained(mae_name).to(device).eval()
    mp = VideoMAEImageProcessor.from_pretrained(mae_name)
    root, out = Path(a.data), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    settings = {'method':a.method,'frames':a.frames,'flow_size':a.flow_size,'gma_iters':a.flow_iters,'precision':a.precision,
                'gma_fp32':a.gma_fp32,'clip_budget':a.clip_budget,
                'clip':clip_name,'mae':mae_name,'scales':[1,2],'flow_checkpoint':Path(a.gma_checkpoint).name}
    config_path = out/'settings.json'
    # Shards must use the same settings. Atomic creation is sufficient since both write identical data.
    if config_path.exists() and json.loads(config_path.read_text()) != settings: raise ValueError('Feature cache settings differ; choose a new output folder')
    tmp = out/f'settings-{a.shard}.tmp'; tmp.write_text(json.dumps(settings)); tmp.replace(config_path)
    rows = read_manifest(root)
    if a.offset: rows = rows[a.offset:]
    if a.limit: rows = rows[:a.limit]
    rows = rows[a.shard::a.shards]

    def bounded(fn, items, label):
        # Retry smaller batches on OOM, without changing frame/clip selection.
        try:
            return fn(items)
        except torch.cuda.OutOfMemoryError:
            if len(items) <= 1:
                raise
        torch.cuda.empty_cache()
        cut = len(items) // 2
        return np.concatenate([bounded(fn, items[:cut], label), bounded(fn, items[cut:], label)], axis=0)

    def tick():
        torch.cuda.synchronize(device)
        return time.perf_counter()

    @torch.inference_mode()
    def features(frames, scores):
        t0 = tick()
        n = len(frames)
        selected = select(scores, a.frames, a.method)
        # Flow is always computed between adjacent original frames, never across skipped frames.
        flow_count = max(n-1, 1)
        clips = windows(flow_count)
        clip_scores = np.array([scores[np.minimum(w+1,n-1)].mean() for w in clips])
        clip_budget = max(1, math.ceil(max(0,a.frames-16)/8)+1)
        if a.clip_budget: clip_budget = min(clip_budget, a.clip_budget)
        chosen = select(clip_scores, clip_budget, a.method)
        clips = [clips[i] for i in chosen]
        needed = np.unique(np.concatenate(clips))
        flow_images = {}
        resized = {}
        def frame(i):
            if i not in resized:
                resized[i] = cv2.resize(frames[i], (a.flow_size,a.flow_size))
            return resized[i]
        if n == 1:
            flow_images[0] = flow_viz.flow_to_image(np.zeros((a.flow_size,a.flow_size,2),np.float32))
        else:
            def flow_batch(ids):
                x = torch.from_numpy(np.stack([frame(i) for i in ids])).permute(0,3,1,2).float().to(device)
                y = torch.from_numpy(np.stack([frame(i+1) for i in ids])).permute(0,3,1,2).float().to(device)
                _, flow = gma(x,y,iters=a.flow_iters,test_mode=True)
                result = flow.permute(0,2,3,1).float().cpu().numpy()
                if not np.isfinite(result).all() and gma.args.mixed_precision:
                    gma.args.mixed_precision = False
                    try:
                        _, flow = gma(x,y,iters=a.flow_iters,test_mode=True)
                        result = flow.permute(0,2,3,1).float().cpu().numpy()
                    finally:
                        gma.args.mixed_precision = True
                if not np.isfinite(result).all(): raise ValueError('Nonfinite optical flow')
                return result
            for start in range(0,len(needed),a.flow_batch):
                ids = needed[start:start+a.flow_batch]
                flows = bounded(flow_batch, ids, 'GMA')
                for i, f in zip(ids, flows):
                    flow_images[i] = flow_viz.flow_to_image(f)
        t1 = tick()
        spatial = []
        def clip_batch(ids):
            imgs = [frames[i] for i in ids]
            x = cp(images=imgs,return_tensors='pt').pixel_values.to(device)
            def run(enabled):
                with torch.autocast('cuda',dtype=torch.float16,enabled=enabled):
                    def forward(x): return clip(pixel_values=x,output_hidden_states=True).hidden_states[-1]
                    return s2.forward(forward,x,scales=[1,2],num_prefix_token=1)[:,0].float().cpu().numpy()
            z = run(fast)
            if fast and not np.isfinite(z).all(): z = run(False)
            return z
        for start in range(0,len(selected),a.clip_batch):
            spatial.append(bounded(clip_batch, selected[start:start+a.clip_batch], 'CLIP'))
        t2 = tick()
        motion = []
        def mae_batch(windows_batch):
            video = [[flow_images[i] for i in w] for w in windows_batch]
            x = mp(images=video,return_tensors='pt').pixel_values.to(device)
            def run(enabled):
                with torch.autocast('cuda',dtype=torch.float16,enabled=enabled):
                    return mae(pixel_values=x,output_hidden_states=True).hidden_states[-1][:,0].float().cpu().numpy()
            z = run(fast)
            if fast and not np.isfinite(z).all(): z = run(False)
            return z
        for start in range(0,len(clips),a.mae_batch):
            motion.append(bounded(mae_batch, clips[start:start+a.mae_batch], 'VideoMAE'))
        t3 = tick()
        timing = {'gma_s':round(t1-t0,2),'clip_s':round(t2-t1,2),'mae_s':round(t3-t2,2),
                  'flow_pairs':0 if n==1 else len(needed),'selected_frames':len(selected),'motion_clips':len(clips)}
        s,m = np.concatenate(spatial),np.concatenate(motion)
        if s.shape != (len(selected),2048) or m.shape != (len(clips),1024): raise ValueError((s.shape,m.shape))
        # Upstream valid-convolution TCN requires at least 16 joint tokens.
        if len(s)+len(m)<16: raise ValueError('Video too short for upstream TCN')
        if not (np.isfinite(s).all() and np.isfinite(m).all()): raise ValueError('Nonfinite features')
        s,m = s.astype(np.float16),m.astype(np.float16)
        if not (np.isfinite(s).all() and np.isfinite(m).all()): raise ValueError('FP16 cache overflow')
        return s,m,selected,np.array(clips),n,timing

    failed = []
    timings = []
    for row in tqdm(rows,desc=f'GPU shard {a.shard}'):
        dest = out/f'{key(row)}.npz'
        if dest.exists(): continue
        try:
            begin = time.perf_counter()
            frames,scores = read_video(root/row['video_file'])
            decoded = time.perf_counter()
            s,m,idx,clips,n,timing = features(frames,scores)
            tmp = dest.with_suffix('.tmp')
            with tmp.open('wb') as f:
                np.savez(f,spatial=s,motion=m,indices=idx,flow_clips=clips,original_frames=n)
            tmp.replace(dest)
            timing.update(id=row['id'],original_frames=n,decode_s=round(decoded-begin,2),total_s=round(time.perf_counter()-begin,2))
            timings.append(timing)
            if a.profile or len(timings) % 50 == 0:
                print('TIMING', json.dumps(timing), flush=True)
        except Exception as e:
            failed.append({'id':row['id'],'error':str(e)})
            print(f'FAILED {row["id"]}: {e}',flush=True)
            if len(failed)>=3: break
    if timings:
        means = {k:round(float(np.mean([x[k] for x in timings])),2) for k in ('decode_s','gma_s','clip_s','mae_s','total_s')}
        print('PROFILE', json.dumps({'count':len(timings), **means}), flush=True)
        with (out/f'timings-{a.shard}.jsonl').open('a') as f:
            for row in timings: f.write(json.dumps(row)+'\n')
    (out/f'failures-{a.shard}.json').write_text(json.dumps(failed,indent=2))
    if failed: raise SystemExit('Extraction incomplete: inspect failure report before training')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    p.add_argument('--spamo',required=True); p.add_argument('--gma',required=True); p.add_argument('--gma-checkpoint',required=True)
    p.add_argument('--method',choices=['all','uniform','motion-aware','adaptive'],default='adaptive')
    p.add_argument('--frames',type=int,default=256); p.add_argument('--flow-size',type=int,default=256)
    p.add_argument('--clip-batch',type=int,default=8); p.add_argument('--flow-batch',type=int,default=8); p.add_argument('--mae-batch',type=int,default=4)
    p.add_argument('--shards',type=int,default=1); p.add_argument('--shard',type=int,default=0); p.add_argument('--limit',type=int,default=0)
    p.add_argument('--precision',choices=['fp16','fp32'],default='fp16')
    p.add_argument('--gma-fp32',action='store_true')
    p.add_argument('--flow-iters',type=int,default=12)
    p.add_argument('--clip-budget',type=int,default=0,help='Optional cap on selected 16-frame motion windows; 0 preserves original logic')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--offset',type=int,default=0)
    a=p.parse_args()
    if min(a.clip_batch,a.flow_batch,a.mae_batch,a.flow_iters)<1 or min(a.clip_budget,a.offset,a.limit)<0: p.error('Invalid batch, iteration count or limit')
    if a.flow_size%8 or a.flow_size<128 or not 0<=a.shard<a.shards or a.frames<16: p.error('Invalid flow resolution, shard or frame budget')
    extractor(a)

if __name__=='__main__': main()
