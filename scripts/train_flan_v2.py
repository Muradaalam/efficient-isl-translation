"""Improved GloSL pose-to-FLAN-T5 trainer and dataset audit.
Run from repository root: python3 -m scripts.train_flan_v2 ...
"""
import argparse
import json
import math
import os
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pose_format import Pose
from torch.utils.data import Dataset
from torch.distributed.nn.functional import all_gather as gather_grad
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF
from jiwer import wer

MODEL = 'google/flan-t5-base'

# MediaPipe face indices: lips, eye contours, brows and nose. New explicit
# subset, not assumed to be identical to the original team's 25-point subset.
FACE25 = [0, 13, 14, 17, 61, 291, 78, 308, 33, 133, 159, 145,
          362, 263, 386, 374, 70, 105, 107, 336, 334, 300, 1, 4, 168]


def read_raw(path):
    with open(path, 'rb') as f:
        pose = Pose.read(f.read())
    data = np.ma.asarray(pose.body.data)
    if data.ndim != 4 or data.shape[1] != 1 or data.shape[-1] < 3:
        raise ValueError(f'Unexpected pose shape {data.shape}: {path}')
    coords = np.asarray(data[:, 0].filled(0), dtype=np.float32)
    conf = np.asarray(pose.body.confidence)[:, 0]
    masked = np.ma.getmaskarray(data[:, 0])
    components, offset = {}, 0
    for component in pose.header.components:
        count = len(component.points)
        components[component.name.upper()] = np.arange(offset, offset + count)
        offset += count
    order = []
    for name, count in [('POSE_LANDMARKS', 33), ('LEFT_HAND_LANDMARKS', 21),
                        ('RIGHT_HAND_LANDMARKS', 21), ('FACE_LANDMARKS', 468)]:
        ix = components.get(name)
        if ix is None or len(ix) != count: raise ValueError(f'Invalid {name}: {path}')
        order.extend(ix)
    coords = coords[:, order, :3]
    valid = (np.isfinite(conf[:, order]) & (conf[:, order] > 0) &
             np.isfinite(coords).all(-1) & ~masked[:, order, :3].any(-1))
    coords[~valid] = 0
    if not valid.any(): raise ValueError(f'No valid landmarks: {path}')
    return coords, valid


def interpolate_anchor(values, valid):
    t = np.arange(len(values))
    if not valid.any(): return np.zeros_like(values)
    return np.stack([np.interp(t, t[valid], values[valid, j])
                     for j in range(values.shape[1])], axis=-1).astype(np.float32)


def normalize_pose(coords, valid):
    p = coords.copy()
    shoulders = valid[:, 11] & valid[:, 12]
    hips = valid[:, 23] & valid[:, 24]
    # Use a single anatomical anchor throughout a recording. Missing anchors
    # are interpolated; use hips only when shoulders are entirely absent.
    if shoulders.any():
        center = interpolate_anchor((p[:, 11, :2]+p[:, 12, :2])/2, shoulders)
    elif hips.any():
        center = interpolate_anchor((p[:, 23, :2]+p[:, 24, :2])/2, hips)
    else:
        # No reliable body anchor: use a fixed robust sequence center.
        center = np.broadcast_to(np.median(p[valid, :2], axis=0), (len(p), 2))
    widths = np.linalg.norm(p[:, 11, :2]-p[:, 12, :2], axis=-1)
    usable = shoulders & np.isfinite(widths) & (widths > 1e-6)
    if usable.any(): scale = float(np.median(widths[usable]))
    else:
        xy = p[valid, :2]
        scale = float(np.linalg.norm(np.percentile(xy, 90, axis=0)-np.percentile(xy, 10, axis=0)))
    scale = max(scale, 1e-5)
    p[:, :, :2] = (p[:, :, :2]-center[:, None, :])/scale
    # MediaPipe regions have different depth origins; interpolate each anchor.
    for a,b,anchor in [(0,33,23),(33,54,33),(54,75,54),(75,543,75)]:
        z = interpolate_anchor(p[:, anchor, 2:3], valid[:, anchor])
        p[:, a:b, 2] = (p[:, a:b, 2]-z)/scale
    p[~valid] = 0
    return p.astype(np.float32)


def sample_pose(pose, valid, frames, method):
    candidates = np.flatnonzero(valid.any(axis=1))
    if len(candidates) == 0: raise ValueError('No valid frames')
    def uniform(n):
        return candidates[np.linspace(0, len(candidates)-1, n).round().astype(int)]
    if len(candidates) <= frames: ix = candidates
    elif method == 'uniform': ix = uniform(frames)
    else:
        hand = pose[:, 33:75]
        pair = valid[1:,33:75] & valid[:-1,33:75]
        displacement = np.linalg.norm(hand[1:]-hand[:-1], axis=-1)
        scores = np.r_[0., (displacement*pair).sum(1)/pair.sum(1).clip(1)]
        if np.allclose(scores[candidates], 0): ix = uniform(frames)
        else:
            base = uniform(frames//2) if method == 'adaptive' else np.empty(0, dtype=int)
            remaining = np.setdiff1d(candidates, base)
            ranked = remaining[np.argsort(-scores[remaining], kind='stable')]
            ix = np.sort(np.r_[base, ranked[:frames-len(base)]])
    n = len(ix)
    p = np.zeros((frames, pose.shape[1], 3), np.float32)
    v = np.zeros((frames, pose.shape[1]), bool)
    indices = np.full(frames, -1, np.int64)
    p[:n], v[:n], indices[:n] = pose[ix], valid[ix], ix
    return p, v, indices


def resolve_path(root, value):
    p = Path(str(value))
    if not p.is_absolute(): p = Path(root)/p
    if not p.is_file():
        # Preserve existing fallback, but refuse if neither path exists.
        p = Path(root)/'poses'/p.name
    if not p.is_file(): raise FileNotFoundError(p)
    return p


class FLANGloSLDataset(Dataset):
    def __init__(self, manifest, pose_root, split, method, target_frames,
                 pose_column, text_column, landmarks=100):
        df = pd.read_csv(manifest)
        self.df = df[df.split == split].reset_index(drop=True)
        self.root, self.method, self.frames = pose_root, method, target_frames
        self.pose_column, self.text_column, self.landmarks = pose_column, text_column, landmarks
        if self.df[[pose_column,text_column]].isna().any().any(): raise ValueError(f'Missing path/text in {split}')
        if not len(self.df): raise ValueError(f'Empty split {split}')

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        pose, valid = read_raw(resolve_path(self.root, row[self.pose_column]))
        pose = normalize_pose(pose, valid)
        if self.landmarks == 100:
            ix = np.r_[np.arange(75), 75+np.asarray(FACE25)]
            pose, valid = pose[:, ix], valid[:, ix]
        p, v, indices = sample_pose(pose, valid, self.frames, self.method)
        return dict(pose=torch.from_numpy(p), valid_mask=torch.from_numpy(v),
                    frame_mask=torch.from_numpy(v.any(1)), indices=torch.from_numpy(indices),
                    sentence=str(row[self.text_column]))


def audit(args):
    df = pd.read_csv(args.manifest)
    paths = [resolve_path(args.pose_root, x).resolve() for x in df[args.pose_column]]
    df = df.assign(resolved=list(map(str, paths)))
    if df[args.text_column].isna().any() or df[args.text_column].astype(str).str.strip().eq('').any():
        raise ValueError('Empty references in manifest')
    if df.groupby('resolved').split.nunique().max() > 1: raise ValueError('Pose path crosses splits')
    if df.resolved.duplicated().any(): print('WARNING: duplicate paths within split')
    rows = []
    for split in ['train','val','test']:
        part = df[df.split == split]
        print(split, 'n=',len(part),'unique texts=',part[args.text_column].nunique(),
              'mean words=',part[args.text_column].str.split().str.len().mean())
        part = part.sample(min(args.audit_samples,len(part)),random_state=args.seed)
        print('Sample references:', part[args.text_column].head(5).tolist())
        for _, row in tqdm(part.iterrows(),total=len(part),desc=split):
            p,v = read_raw(row.resolved)
            q = normalize_pose(p,v)
            selected = v if args.landmarks == 543 else v[:, np.r_[np.arange(75), 75+np.asarray(FACE25)]]
            n = int(selected.any(1).sum())
            if n == 0: raise ValueError(f'No valid selected landmarks: {row.resolved}')
            rows.append(dict(split=split, path=row.resolved, frames=len(p), valid_frames=n,
                hand_valid=float(v[:,33:75].mean()), shoulder_valid=float((v[:,11]&v[:,12]).mean()),
                hip_valid=float((v[:,23]&v[:,24]).mean()), coordinate_p99=float(np.percentile(np.abs(q[v]),99)),
                coordinate_max=float(np.abs(q).max())))
    stats = pd.DataFrame(rows)
    print(stats.groupby('split').mean(numeric_only=True).to_string())
    train = stats[stats.split=='train'].valid_frames
    print('Training sample compression estimates:')
    for b in [32,64,128,256]:
        print(b, 'compressed fraction=', round(float((train>b).mean()),3),
              'frame reduction=',round(float(1-np.minimum(train,b).sum()/train.sum()),3))
    # Choose largest of 64/128 giving at least half of sampled training clips compression.
    budget = 128 if (train>128).mean() >= .5 else 64
    print('Suggested starting budget:',budget)
    out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    stats.to_csv(out/'audit.csv',index=False)
    write_json(out/'audit_summary.json',{'suggested_frames':budget,'sample_count':len(stats)})



class PoseTranslator(nn.Module):
    def __init__(self, model_name=MODEL, landmarks=100, contrastive_weight=.5, prefix_weight=3., prefix_tokens=3):
        super().__init__()
        # Use the same loading path as the previous working baseline. Do not
        # override tie_word_embeddings at load time: some Transformers versions
        # disable encoder/decoder input sharing as well when it is passed.
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
        shared = self.t5.get_input_embeddings()
        # T5 encoder and decoder must both use the pretrained shared input table.
        # This does not tie the separate FLAN-T5 language-model output head.
        self.t5.set_input_embeddings(shared)
        for name, stack in [('encoder', self.t5.get_encoder()), ('decoder', self.t5.get_decoder())]:
            if stack.get_input_embeddings().weight is not shared.weight:
                raise RuntimeError(f'{name} input embeddings are not shared; stop before training.')
        if self.t5.get_output_embeddings().weight is shared.weight:
            raise RuntimeError('Unexpected tied LM head for FLAN-T5; stop before training.')
        print('Verified shared encoder/decoder input embeddings and separate LM head.', flush=True)
        self.contrastive_weight = contrastive_weight
        self.prefix_weight = prefix_weight
        self.prefix_tokens = prefix_tokens
        d = self.t5.config.d_model
        self.landmarks = landmarks
        # Position, displacement, velocity, two validity indicators per landmark, plus two time features.
        self.spatial = nn.Sequential(nn.Linear(landmarks * 11 + 2, d), nn.LayerNorm(d), nn.GELU())
        self.temporal = nn.ModuleList([nn.Conv1d(d, d, 5, stride=2, padding=2) for _ in range(2)])
        self.norm = nn.LayerNorm(d)
        with torch.no_grad():
            norms = self.t5.get_input_embeddings().weight.float().norm(dim=-1)
            token_norm = norms.mean().item()
        self.initial_scale = token_norm / math.sqrt(d)
        # Log parameter ensures positive scale throughout training.
        self.log_scale = nn.Parameter(torch.tensor(math.log(max(self.initial_scale, 1e-6))))
        print(f'T5 hidden={d}, text norm={token_norm:.3f}, initial scale={self.initial_scale:.3f}', flush=True)

    def encode_pose(self, pose, valid_mask, frame_mask, indices):
        if pose.ndim != 4 or pose.shape[-2:] != (self.landmarks, 3):
            raise ValueError(f'Expected B,T,{self.landmarks},3; got {pose.shape}')
        mask = frame_mask.bool()
        valid = valid_mask.bool() & mask.unsqueeze(-1) & torch.isfinite(pose).all(-1)
        if not mask.any(dim=1).all():
            raise ValueError('Encountered a sequence with no valid selected frames')
        p = torch.where(valid.unsqueeze(-1), pose.float(), 0.)
        delta = torch.zeros_like(p)
        pair = valid[:, 1:] & valid[:, :-1]
        delta[:, 1:] = torch.where(pair.unsqueeze(-1), p[:, 1:] - p[:, :-1], 0.)
        gap = torch.zeros_like(indices, dtype=p.dtype)
        gap[:, 1:] = (indices[:, 1:] - indices[:, :-1]).clamp_min(1).float()
        gap = gap * mask
        velocity = delta / gap.clamp_min(1)[:, :, None, None]
        pair_mask = torch.zeros_like(valid)
        pair_mask[:, 1:] = pair
        relative = indices.clamp_min(0).float() / indices.max(dim=1, keepdim=True).values.clamp_min(1)
        features = torch.cat((p.flatten(2), delta.flatten(2), velocity.flatten(2),
                              valid.float(), pair_mask.float(), gap.log1p().unsqueeze(-1),
                              relative.unsqueeze(-1)), dim=-1)
        x = self.spatial(features) * mask.unsqueeze(-1)
        for conv in self.temporal:
            x = F.gelu(conv(x.transpose(1, 2)).transpose(1, 2))
            # Match convolution receptive field exactly, including internal holes.
            mask = F.max_pool1d(mask.float().unsqueeze(1), kernel_size=5, stride=2, padding=2).squeeze(1).bool()
            x = x * mask.unsqueeze(-1)
        x = self.norm(x) * self.log_scale.exp()
        x = x * mask.unsqueeze(-1)
        return x, mask.long()

    @staticmethod
    def pooled(hidden, mask):
        mask = mask.unsqueeze(-1).to(hidden.dtype)
        return F.normalize((hidden * mask).sum(1) / mask.sum(1).clamp_min(1), dim=-1)

    def alignment_loss(self, hidden, mask, labels):
        text_mask = labels.ne(-100)
        text_ids = labels.masked_fill(~text_mask, self.t5.config.pad_token_id)
        # Stop-gradient text targets; the encoder is shared with the translation branch.
        # Disable dropout for the target branch only, restoring the previous mode.
        encoder = self.t5.get_encoder()
        was_training = encoder.training
        encoder.eval()
        try:
            with torch.no_grad():
                text_hidden = encoder(input_ids=text_ids, attention_mask=text_mask.long(), return_dict=True).last_hidden_state
                text_vec = self.pooled(text_hidden, text_mask)
        finally:
            encoder.train(was_training)
        pose_vec = self.pooled(hidden, mask)
        if self.training and dist.is_initialized() and dist.get_world_size() > 1:
            world = dist.get_world_size()
            width = torch.tensor(labels.shape[1], device=labels.device)
            dist.all_reduce(width, op=dist.ReduceOp.MAX)
            labels = F.pad(labels, (0, int(width.item())-labels.shape[1]), value=-100)
            def gather(x):
                parts = [torch.empty_like(x) for _ in range(world)]
                dist.all_gather(parts, x.contiguous())
                return torch.cat(parts)
            global_labels = gather(labels)
            global_text = gather(text_vec)
            global_pose = torch.cat(gather_grad(pose_vec.contiguous()))
        else:
            global_labels, global_text, global_pose = labels, text_vec, pose_vec
        positives = (labels[:, None, :] == global_labels[None, :, :]).all(-1)
        def multi_positive(logits, positive):
            return (torch.logsumexp(logits, dim=1) - torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)).mean()
        return .5 * (multi_positive(pose_vec @ global_text.T / .07, positives) +
                     multi_positive(text_vec @ global_pose.T / .07, positives))

    def forward(self, pose, valid_mask, frame_mask, indices, labels, details=False):
        x, mask = self.encode_pose(pose, valid_mask, frame_mask, indices)
        encoded = self.t5.get_encoder()(inputs_embeds=x, attention_mask=mask, return_dict=True)
        output = self.t5(encoder_outputs=encoded, attention_mask=mask, labels=labels, use_cache=False)
        ce = output.loss
        if details:
            valid = labels.ne(-100)
            pred = output.logits.argmax(-1)
            return ce, pred, valid
        if not self.training:
            return ce
        # Normalize by token weights to keep the objective scale comparable.
        per_token = F.cross_entropy(output.logits.transpose(1, 2), labels, reduction='none', ignore_index=-100)
        weights = labels.ne(-100).to(per_token.dtype)
        weights[:, :self.prefix_tokens] *= self.prefix_weight
        loss = (per_token * weights).sum() / weights.sum().clamp_min(1)
        contrastive = loss.new_zeros(())
        if self.contrastive_weight > 0:
            contrastive = self.alignment_loss(encoded.last_hidden_state, mask, labels)
        self.last_losses = torch.stack((ce.detach(), loss.detach(), contrastive.detach()))
        return loss + self.contrastive_weight * contrastive

    @torch.no_grad()
    def translate(self, pose, valid_mask, frame_mask, indices, beams=4, max_tokens=96):
        x, mask = self.encode_pose(pose, valid_mask, frame_mask, indices)
        encoded = self.t5.get_encoder()(inputs_embeds=x, attention_mask=mask, return_dict=True)
        return self.t5.generate(encoder_outputs=encoded, attention_mask=mask, num_beams=beams,
                                max_new_tokens=max_tokens, do_sample=False, use_cache=False)


class Collator:
    def __init__(self, tokenizer, max_tokens):
        self.tokenizer, self.max_tokens = tokenizer, max_tokens

    def __call__(self, items):
        batch = {k: torch.stack([torch.as_tensor(item[k]) for item in items])
                 for k in ('pose', 'valid_mask', 'frame_mask', 'indices')}
        sentences = [str(item['sentence']) for item in items]
        text = self.tokenizer(sentences, padding=True, truncation=True,
                              max_length=self.max_tokens, return_tensors='pt')
        batch['labels'] = text.input_ids.masked_fill(text.attention_mask == 0, -100)
        batch['sentences'] = sentences
        return batch


def move(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}


def norm_text(s):
    return ' '.join(s.lower().split())


def metrics(preds, refs):
    p, r = list(map(norm_text, preds)), list(map(norm_text, refs))
    return {'bleu4': BLEU(effective_order=True).corpus_score(p, [r]).score,
            'chrf': CHRF().corpus_score(p, [r]).score,
            'wer': wer(r, p), 'exact_match': sum(a == b for a, b in zip(p, r)) / len(r),
            'blank_rate': sum(not a for a in p) / len(p)}


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, args, preview=4, shuffled=False):
    model.eval()
    preds, refs = [], []
    losses, tokens = 0., 0
    first_correct, first_count = 0, 0
    for batch in loader:
        x = move(batch, device)
        if shuffled:
            if len(x['pose']) == 1:
                continue
            for key in ('pose', 'valid_mask', 'frame_mask', 'indices'):
                x[key] = x[key].roll(1, dims=0)
        n = int((x['labels'] != -100).sum().item())
        ce, tf_pred, valid = model(**x, details=True)
        losses += ce.item() * n
        first_correct += int(((tf_pred[:, 0] == x['labels'][:, 0]) & valid[:, 0]).sum())
        first_count += int(valid[:, 0].sum())
        tokens += n
        ids = model.translate(x['pose'], x['valid_mask'], x['frame_mask'], x['indices'], args.beams, args.max_tokens)
        preds.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
        refs.extend(batch['sentences'])
    result = metrics(preds, refs)
    result['loss'] = losses / max(tokens, 1)
    result['first_token_accuracy'] = first_correct / max(first_count, 1)
    result['unique_predictions'] = len(set(map(norm_text, preds)))
    for p, r in list(zip(preds, refs))[:preview]:
        print(f'GT: {r}\nPR: {p!r}', flush=True)
    print(json.dumps(result), flush=True)
    return result, [{'reference': r, 'prediction': p} for p, r in zip(preds, refs)]


@torch.no_grad()
def text_check(model, tokenizer, device):
    model.eval()
    inputs = tokenizer('Translate to German: I am going to school.', return_tensors='pt').to(device)
    for cache in (True, False):
        ids = model.t5.generate(**inputs, max_new_tokens=32, num_beams=4, use_cache=cache)
        pred = tokenizer.decode(ids[0], skip_special_tokens=True)
        print(f'Pretrained text check cache={cache}: {pred!r}, ids={ids[0].tolist()}', flush=True)
        # The previous working checkpoint generated German containing Schule.
        # Fail closed on gibberish as well as blank predictions, before any update.
        if 'schule' not in pred.casefold() or len(pred.split()) < 3:
            raise RuntimeError('Pretrained text smoke test failed; stop before training. Inspect model loading and Transformers version.')


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def make_dataset(args, split):
    return FLANGloSLDataset(manifest=args.manifest, pose_root=args.pose_root, split=split,
                            method=args.method, target_frames=args.frames,
                            pose_column=args.pose_column, text_column=args.text_column, landmarks=args.landmarks)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--pose-root', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--pose-column', default='pose_file')
    ap.add_argument('--text-column', default='text')
    ap.add_argument('--method', default='adaptive', choices=['adaptive', 'uniform', 'motion-aware'])
    ap.add_argument('--frames', type=int, default=128)
    ap.add_argument('--landmarks', type=int, choices=[100,543], default=100)
    ap.add_argument('--audit-samples', type=int, default=100)
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--mode', choices=['audit', 'overfit', 'train', 'test'], default='train')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--cnn-lr', type=float, default=5e-4)
    ap.add_argument('--t5-lr', type=float, default=5e-5)
    ap.add_argument('--warmup', type=float, default=.01)
    ap.add_argument('--max-tokens', type=int, default=96)
    ap.add_argument('--beams', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--contrastive-weight', type=float, default=.5)
    ap.add_argument('--prefix-weight', type=float, default=3.)
    ap.add_argument('--prefix-tokens', type=int, default=3)
    ap.add_argument('--overfit-samples', type=int, default=32)
    ap.add_argument('--overfit-steps', type=int, default=200)
    ap.add_argument('--check-every', type=int, default=25)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--checkpoint', default=None)
    args = ap.parse_args()
    if args.frames < 2 or args.batch_size < 1 or args.audit_samples < 1: ap.error('frames >= 2, batch size >= 1, audit samples >= 1 required')
    if args.mode == 'audit':
        audit(args)
        return
    rank, world = int(os.getenv('RANK', 0)), int(os.getenv('WORLD_SIZE', 1))
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    if world > 1 and args.mode != 'train':
        raise ValueError('Use one GPU for overfit/test mode.')
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(hours=1))
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    out = Path(args.output)
    if args.mode in ('train', 'overfit') and not args.resume and (out / 'last.pt').exists():
        raise FileExistsError(f'Existing checkpoint in {out}; choose a new output or use --resume for full training')
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = PoseTranslator(args.model, landmarks=args.landmarks, contrastive_weight=args.contrastive_weight, prefix_weight=args.prefix_weight, prefix_tokens=args.prefix_tokens).to(device)
    if args.mode != 'test':
        text_check(model, tokenizer, device)
        if rank == 0:
            print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
    collator = Collator(tokenizer, args.max_tokens)

    def loader(dataset, shuffle=False, sampler=None, workers=None):
        n = args.workers if workers is None else workers
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
                          num_workers=n, collate_fn=collator, pin_memory=device.type == 'cuda',
                          persistent_workers=n > 0)

    if args.mode == 'test':
        checkpoint = Path(args.checkpoint or out / 'best.pt')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'])
        for key in ('model', 'method', 'frames', 'landmarks'):
            if state['args'][key] != getattr(args, key):
                raise ValueError(f'Checkpoint {key}={state["args"][key]} but requested {getattr(args, key)}')
        test = make_dataset(args, 'test')
        results, predictions = evaluate(model, loader(test), tokenizer, device, args, preview=10)
        write_json(out / 'test_metrics.json', results)
        write_json(out / 'test_predictions.json', predictions)
        if args.batch_size > 1 and len(test) > 1:
            # If final batch has one item, omit it from both comparison scores.
            count = len(test) - int(len(test) % args.batch_size == 1)
            subset = Subset(test, range(count))
            print('Shuffled-pose grounding control:', flush=True)
            shuffled, control_predictions = evaluate(model, loader(subset), tokenizer, device, args, shuffled=True)
            correct = metrics([p['prediction'] for p in predictions[:count]], [p['reference'] for p in predictions[:count]])
            write_json(out / 'grounding_control.json', {'count': count, 'correct': correct, 'shuffled': shuffled})
            write_json(out / 'shuffled_predictions.json', control_predictions)
        return

    train = make_dataset(args, 'train')
    if args.mode == 'overfit':
        # Cache only 32 training examples; repeated pose decoding is unnecessary.
        indices = random.Random(args.seed).sample(range(len(train)), min(args.overfit_samples, len(train)))
        train = [train[i] for i in indices]
        train_loader = loader(train, shuffle=True, workers=0)
        val_loader = loader(train, workers=0)
        print('Overfit on training samples only:', len(train), flush=True)
        example = next(iter(val_loader))
        with torch.no_grad():
            x = move(example, device)
            embeds, mask = model.encode_pose(x['pose'], x['valid_mask'], x['frame_mask'], x['indices'])
            print('Pose shape:', tuple(x['pose'].shape), 'T5 inputs:', tuple(embeds.shape),
                  'Embedding norm:', embeds[mask.bool()].norm(dim=-1).mean().item(), flush=True)
    else:
        sampler = DistributedSampler(train, shuffle=True, seed=args.seed) if world > 1 else None
        train_loader = loader(train, shuffle=sampler is None, sampler=sampler)
        if rank == 0:
            val = make_dataset(args, 'val')
            print(f'Train: {len(train)}, val: {len(val)}', flush=True)
            val_loader = loader(val)

    optimizer = torch.optim.AdamW([
        {'params': model.t5.parameters(), 'lr': args.t5_lr},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('t5.')], 'lr': args.cnn_lr}
    ], weight_decay=.01)
    total = args.overfit_steps if args.mode == 'overfit' else args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total * args.warmup), total)
    start, best, best_loss, stale, history = 0, -1., float('inf'), 0, []
    if args.resume:
        if args.mode != 'train': raise ValueError('--resume is for full training')
        state = torch.load(out / 'last.pt', map_location='cpu', weights_only=True)
        for key in ('model', 'method', 'frames', 'landmarks', 'epochs', 'batch_size', 'contrastive_weight', 'prefix_weight', 'prefix_tokens'):
            if state['args'][key] != getattr(args, key): raise ValueError(f'Resume mismatch: {key}')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start, best, best_loss, stale, history = (state[k] for k in ('epoch', 'best', 'best_loss', 'stale', 'history'))
        del state
    wrapped = DDP(model, device_ids=[local_rank]) if world > 1 else model

    def update(batch):
        wrapped.train()
        optimizer.zero_grad(set_to_none=True)
        loss = wrapped(**move(batch, device))
        if not torch.isfinite(loss): raise RuntimeError(f'Non-finite loss: {loss.item()}')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        return loss.detach().item()

    if args.mode == 'overfit':
        steps = 0
        best_bleu = -1.0
        while steps < total:
            for batch in train_loader:
                loss = update(batch)
                steps += 1
                if steps % args.check_every == 0 or steps == total:
                    print(f'Overfit step {steps}/{total}, batch loss={loss:.4f}', flush=True)
                    result, preds = evaluate(model, val_loader, tokenizer, device, args)
                    history.append({'step': steps, **result})
                    save(out / 'last.pt', {'model': model.state_dict(), 'args': vars(args), 'step': steps})
                    if result['exact_match'] > best or (result['exact_match'] == best and result['bleu4'] > best_bleu):
                        best, best_bleu = result['exact_match'], result['bleu4']
                        save(out / 'best.pt', {'model': model.state_dict(), 'args': vars(args), 'step': steps})
                    write_json(out / 'overfit_history.json', history)
                    write_json(out / 'overfit_predictions.json', preds)
                    if result['exact_match'] >= 1.0 or steps == total:
                        print('Shuffled training-pose control:', flush=True)
                        shuffled, _ = evaluate(model, val_loader, tokenizer, device, args, shuffled=True)
                        write_json(out / 'overfit_grounding.json', {'correct': result, 'shuffled': shuffled})
                    if result['exact_match'] >= 1.0:
                        print('Overfit target reached. This demonstrates train-set memorization, not generalization.', flush=True)
                        return
                if steps >= total: break
        print('Overfit run completed; inspect generated text and metrics before full training.', flush=True)
        return

    for epoch in range(start, args.epochs):
        if world > 1: sampler.set_epoch(epoch)
        stats = torch.zeros(6, dtype=torch.float64, device=device)
        bar = tqdm(train_loader, desc=f'Epoch {epoch+1}', disable=rank != 0)
        for batch in bar:
            loss = update(batch)
            n = int((batch['labels'] != -100).sum())
            ce, weighted, contrastive = model.last_losses.to(torch.float64).unbind()
            stats += torch.stack((ce*n, ce.new_tensor(n), weighted, contrastive, ce.new_tensor(loss), ce.new_tensor(1)))
            if rank == 0: bar.set_postfix(loss=f'{loss:.3f}')
        if world > 1: dist.all_reduce(stats)
        stop = torch.zeros(1, device=device, dtype=torch.int32)
        if rank == 0:
            result, preds = evaluate(model, val_loader, tokenizer, device, args)
            result.update(epoch=epoch+1, train_ce=(stats[0]/stats[1]).item(), train_weighted_ce=(stats[2]/stats[5]).item(), train_contrastive=(stats[3]/stats[5]).item(), train_objective=(stats[4]/stats[5]).item(), scale=model.log_scale.exp().item())
            history.append(result)
            write_json(out / 'history.json', history)
            improved = result['bleu4'] > best or (result['bleu4'] == best and result['loss'] < best_loss)
            if improved:
                best, best_loss, stale = result['bleu4'], result['loss'], 0
                save(out / 'best.pt', {'model': model.state_dict(), 'args': vars(args), 'epoch': epoch+1})
                write_json(out / 'best_val_predictions.json', preds)
            else: stale += 1
            save(out / 'last.pt', {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                    'scheduler': scheduler.state_dict(), 'args': vars(args), 'epoch': epoch+1,
                                    'best': best, 'best_loss': best_loss, 'stale': stale, 'history': history})
            print(f'Epoch {epoch+1}: {json.dumps(result)}, patience={stale}/{args.patience}', flush=True)
            stop.fill_(int(stale >= args.patience))
        if world > 1: dist.broadcast(stop, src=0)
        if stop.item(): break
    if world > 1: dist.destroy_process_group()


if __name__ == '__main__':
    main()
