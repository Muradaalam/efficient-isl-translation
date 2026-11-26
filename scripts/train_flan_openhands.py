"""OpenHands INCLUDE ST-GCN + FLAN-T5 continuous translation.
Uses the V4 trainer and official pretrained graph implementation.
"""
import argparse
import importlib
import sys
import types
from omegaconf import OmegaConf
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

MODEL = 'google/flan-t5-small'

# MediaPipe face indices: lips, eye contours, brows and nose. New explicit
# subset, not assumed to be identical to the original team's 25-point subset.
FACE25 = [0, 13, 14, 17, 61, 291, 78, 308, 33, 133, 159, 145,
          362, 263, 386, 374, 70, 105, 107, 336, 334, 300, 1, 4, 168]


def read_raw(path, return_dimensions=False):
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
    if return_dimensions:
        return coords, valid, (float(pose.header.dimensions.width), float(pose.header.dimensions.height))
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


OPENHANDS27 = np.array([0,2,5,11,12,13,14,33,37,38,41,42,45,46,49,50,53,54,58,59,62,63,66,67,70,71,74])


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
        raw, valid, dims = read_raw(resolve_path(self.root, row[self.pose_column]), return_dimensions=True)
        if not np.isfinite(dims).all() or min(dims) <= 0:
            raise ValueError(f'Invalid pose dimensions: {dims}')
        # Preserve existing V4 selection policy before adapting coordinates.
        ix = np.r_[np.arange(75), 75+np.asarray(FACE25)]
        _, _, indices = sample_pose(normalize_pose(raw, valid)[:, ix], valid[:, ix], self.frames, self.method)
        xy = raw[:, OPENHANDS27, :2] / np.asarray(dims, dtype=np.float32)
        v = valid[:, OPENHANDS27]
        shoulders = v[:, 3] & v[:, 4]
        widths = np.linalg.norm(xy[:, 3]-xy[:, 4], axis=-1)
        shoulders &= widths > 1e-6
        # Clip-level shoulder normalization; unlike the upstream transform,
        # missing shoulders are excluded rather than treated as observations.
        if shoulders.any():
            center = ((xy[shoulders, 3]+xy[shoulders, 4])/2).mean(0)
            scale = widths[shoulders].mean()
        else:
            if not v.any(): raise ValueError(f'No OpenHands landmarks: {row[self.pose_column]}')
            pts = xy[v]
            center = np.median(pts, axis=0)
            scale = np.linalg.norm(np.percentile(pts,90,axis=0)-np.percentile(pts,10,axis=0))
        xy = ((xy-center)/max(float(scale),1e-6)).astype(np.float32)
        xy[~v] = 0
        selected = indices[indices >= 0]
        selected = selected[v[selected].any(1)]
        if not len(selected):
            candidates = np.flatnonzero(v.any(1))
            if not len(candidates): raise ValueError(f'No OpenHands landmarks: {row[self.pose_column]}')
            selected = candidates[np.linspace(0,len(candidates)-1,min(self.frames,len(candidates))).round().astype(int)]
        n = len(selected)
        p = np.zeros((self.frames,27,2),np.float32)
        mask = np.zeros((self.frames,27),bool)
        idx = np.full(self.frames,-1,np.int64)
        p[:n], mask[:n], idx[:n] = xy[selected], v[selected], selected
        return dict(pose=torch.from_numpy(p), valid_mask=torch.from_numpy(mask),
                    frame_mask=torch.from_numpy(mask.any(1)), indices=torch.from_numpy(idx),
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
            selected = v[:, OPENHANDS27]
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



def load_stgcn(root, config, checkpoint, trusted=False):
    # Load the official graph module without importing OpenHands' old trainer stack.
    graph_dir = Path(root)/'openhands/models/encoder/graph'
    if not (graph_dir/'st_gcn.py').is_file(): raise FileNotFoundError(graph_dir/'st_gcn.py')
    pkg = types.ModuleType('_local_openhands_graph')
    pkg.__path__ = [str(graph_dir.resolve())]
    sys.modules.setdefault(pkg.__name__, pkg)
    STGCN = importlib.import_module(pkg.__name__+'.st_gcn').STGCN
    cfg = OmegaConf.load(config)
    params = cfg.model.encoder.params
    if cfg.model.encoder.type != 'st-gcn' or params.graph_args.num_nodes != 27:
        raise ValueError('Expected INCLUDE ST-GCN 27-point configuration')
    model = STGCN(in_channels=2, **params)
    # Only opt in to legacy pickle loading for a verified official release.
    state = torch.load(checkpoint, map_location='cpu', weights_only=not trusted)
    sd = state.get('state_dict', state)
    expected = model.state_dict()
    suffix = 'data_bn.weight'
    prefixes = [k[:-len(suffix)] for k in sd if k.endswith(suffix)]
    matches = []
    for prefix in prefixes:
        selected = {k[len(prefix):]:v for k,v in sd.items() if k.startswith(prefix)}
        selected = {k:v for k,v in selected.items() if k in expected}
        if set(selected)==set(expected) and all(selected[k].shape==expected[k].shape for k in expected):
            matches.append((prefix,selected))
    if len(matches)!=1:
        raise RuntimeError(f'No unique complete ST-GCN weight match; prefixes={prefixes}, matches={len(matches)}')
    prefix, selected = matches[0]
    if not torch.allclose(selected['A'].float(), expected['A'].float()):
        raise ValueError('Checkpoint graph differs from configuration')
    model.load_state_dict(selected, strict=True)
    model.requires_grad_(False).eval()
    print(f'Loaded pretrained ST-GCN: {len(selected)} tensors, prefix={prefix!r}', flush=True)
    return model


class PoseTranslator(nn.Module):
    def __init__(self, model_name=MODEL, landmarks=27, contrastive_weight=.5, prefix_weight=3., prefix_tokens=3, graph=None):
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
        if landmarks != 27 or graph is None:
            raise ValueError('OpenHands requires the loaded 27-landmark encoder')
        self.freeze_t5 = False
        self.graph = graph
        self.projection = nn.Sequential(nn.Linear(graph.n_out_features+2,d),nn.LayerNorm(d),nn.GELU(),nn.Linear(d,d))
        self.norm = nn.LayerNorm(d)
        with torch.no_grad():
            norms = self.t5.get_input_embeddings().weight.float().norm(dim=-1)
            token_norm = norms.mean().item()
        self.initial_scale = token_norm / math.sqrt(d)
        # Log parameter ensures positive scale throughout training.
        self.log_scale = nn.Parameter(torch.tensor(math.log(max(self.initial_scale, 1e-6))))
        print(f'T5 hidden={d}, text norm={token_norm:.3f}, initial scale={self.initial_scale:.3f}', flush=True)

    def set_frozen_t5(self, freeze):
        self.freeze_t5 = bool(freeze)
        self.t5.requires_grad_(not freeze)
        if freeze: self.t5.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_t5: self.t5.eval()
        self.graph.eval()
        return self

    def encode_pose(self, pose, valid_mask, frame_mask, indices):
        if pose.ndim != 4 or pose.shape[-2:] != (27,2):
            raise ValueError(f'Expected B,T,27,2: {pose.shape}')
        mask = frame_mask.bool() & valid_mask.any(-1)
        n = mask.sum(1)
        if not (n>0).all(): raise ValueError('Empty selected sequence')
        # Replicate the final observed frame in the padded tail for the graph;
        # the T5 attention mask excludes output positions beyond real length.
        p = torch.where(valid_mask.unsqueeze(-1),pose.float(),0.)
        last = p[torch.arange(len(p),device=p.device),n-1]
        p = torch.where(mask[:,:,None,None],p,last[:,None])
        cache = []
        def capture(module, inputs, output):
            cache.append(output[0])
        handle = self.graph.st_gcn_networks[-1].register_forward_hook(capture)
        try:
            with torch.no_grad(): self.graph(p.permute(0,3,1,2).contiguous())
        finally:
            handle.remove()
        if len(cache)!=1: raise RuntimeError('Could not extract graph temporal feature map')
        feat = cache[0].mean(-1).transpose(1,2).contiguous()
        if feat.shape[1] != (p.shape[1]+3)//4: raise ValueError(f'Unexpected graph output: {feat.shape}')
        keep = torch.arange(feat.shape[1],device=p.device)[None,:] < ((n+3)//4)[:,None]
        pos = indices[:,::4][:,:feat.shape[1]].clamp_min(0).float()
        rel = pos/indices.max(1,keepdim=True).values.clamp_min(1)
        gap = torch.zeros_like(pos)
        gap[:,1:] = (pos[:,1:]-pos[:,:-1]).clamp_min(0).log1p()
        x = self.projection(torch.cat((feat,rel.unsqueeze(-1),gap.unsqueeze(-1)),dim=-1))
        return self.norm(x)*self.log_scale.exp()*keep.unsqueeze(-1),keep.long()

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
    return {**{f'bleu{n}': BLEU(max_ngram_order=n, effective_order=True).corpus_score(p, [r]).score for n in range(1,5)},
            'chrf': CHRF().corpus_score(p, [r]).score,
            'wer': wer(r, p), 'exact_match': sum(a == b for a, b in zip(p, r)) / len(r),
            'blank_rate': sum(not a for a in p) / len(p)}


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, args, preview=4, shuffled=False):
    model.eval()
    preds, refs = [], []
    losses, tokens = 0., 0
    first_correct, first_count = 0, 0
    grounding_correct, grounding_wrong, grounding_count = 0., 0., 0
    for batch_idx, batch in enumerate(loader):
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
        if not shuffled and batch_idx < args.probe_batches and len(x['pose']) > 1:
            labels = x['labels']
            # Ignore pairs with identical tokenized references.
            different = labels.ne(labels.roll(1, 0)).any(1)
            if different.any():
                pose_keys = ('pose', 'valid_mask', 'frame_mask', 'indices')
                def nll(roll):
                    inputs = [x[k].roll(1, 0) if roll else x[k] for k in pose_keys]
                    h, m = model.encode_pose(*inputs)
                    enc = model.t5.get_encoder()(inputs_embeds=h, attention_mask=m, return_dict=True)
                    logits = model.t5(encoder_outputs=enc, attention_mask=m, labels=labels.contiguous(), use_cache=False).logits
                    ce = F.cross_entropy(logits.transpose(1,2), labels, reduction='none', ignore_index=-100)
                    return ce.sum(1)/labels.ne(-100).sum(1).clamp_min(1)
                grounding_correct += nll(False)[different].sum().item()
                grounding_wrong += nll(True)[different].sum().item()
                grounding_count += int(different.sum())
        ids = model.translate(x['pose'], x['valid_mask'], x['frame_mask'], x['indices'], args.beams, args.max_tokens)
        preds.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
        refs.extend(batch['sentences'])
    result = metrics(preds, refs)
    result['loss'] = losses / max(tokens, 1)
    result['first_token_accuracy'] = first_correct / max(first_count, 1)
    result['unique_predictions'] = len(set(map(norm_text, preds)))
    result['grounding_pairs'] = grounding_count
    if grounding_count:
        result['correct_pose_ce'] = grounding_correct/grounding_count
        result['shuffled_pose_ce'] = grounding_wrong/grounding_count
        result['grounding_gap'] = (grounding_wrong-grounding_correct)/grounding_count
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
    ap.add_argument('--landmarks', type=int, choices=[27], default=27)
    ap.add_argument('--audit-samples', type=int, default=100)
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--mode', choices=['audit', 'overfit', 'train', 'test'], default='train')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--patience', type=int, default=0)
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
    ap.add_argument('--freeze-t5', action='store_true')
    ap.add_argument('--init-checkpoint', default=None, help='Warm start from compatible OpenHands translation weights only')
    ap.add_argument('--probe-batches', type=int, default=12)
    ap.add_argument('--openhands-root', required=True)
    ap.add_argument('--openhands-config', required=True)
    ap.add_argument('--openhands-checkpoint', required=True)
    ap.add_argument('--trusted-official-checkpoint', action='store_true')
    args = ap.parse_args()
    if args.init_checkpoint and (args.resume or args.mode != 'train'):
        ap.error('--init-checkpoint requires a fresh train run')
    if args.probe_batches < 0: ap.error('probe-batches must be nonnegative')
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
    if args.mode in ('train', 'overfit') and not args.resume and ((out / 'last.pt').exists() or (out / 'best.pt').exists()):
        raise FileExistsError(f'Existing checkpoint in {out}; choose a new output or use --resume for full training')
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    graph = load_stgcn(args.openhands_root, args.openhands_config, args.openhands_checkpoint, args.trusted_official_checkpoint)
    model = PoseTranslator(args.model, landmarks=args.landmarks, contrastive_weight=args.contrastive_weight, prefix_weight=args.prefix_weight, prefix_tokens=args.prefix_tokens, graph=graph).to(device)
    if args.mode != 'test':
        text_check(model, tokenizer, device)
        if rank == 0:
            print('Total parameters:', sum(p.numel() for p in model.parameters()), flush=True)
    model.set_frozen_t5(args.freeze_t5)
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
        for key in ('model', 'method', 'frames', 'landmarks'):
            if state['args'][key] != getattr(args, key): raise ValueError(f'Warm-start mismatch: {key}')
        model.load_state_dict(state['model'], strict=True)
        print('Loaded OpenHands translation weights:', args.init_checkpoint, flush=True)
        del state
    if rank == 0:
        print('Frozen T5:', args.freeze_t5, 'Trainable:', sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
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
        {'params': [p for p in model.t5.parameters() if p.requires_grad], 'lr': args.t5_lr},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('t5.') and p.requires_grad], 'lr': args.cnn_lr}
    ], weight_decay=.01)
    total = args.overfit_steps if args.mode == 'overfit' else args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total * args.warmup), total)
    start, best, best_loss, stale, history = 0, -1., float('inf'), 0, []
    if args.resume:
        if args.mode != 'train': raise ValueError('--resume is for full training')
        state = torch.load(out / 'last.pt', map_location='cpu', weights_only=True)
        for key in ('model', 'method', 'frames', 'landmarks', 'epochs', 'batch_size', 'contrastive_weight', 'prefix_weight', 'prefix_tokens', 'freeze_t5', 'cnn_lr', 't5_lr', 'warmup', 'seed'):
            if state['args'][key] != getattr(args, key): raise ValueError(f'Resume mismatch: {key}')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start, best, best_loss, stale, history = (state[k] for k in ('epoch', 'best', 'best_loss', 'stale', 'history'))
        del state
    # Preserve the initial validation result for phase-2 warm starts.
    if args.init_checkpoint and rank == 0:
        result, preds = evaluate(model, val_loader, tokenizer, device, args)
        best, best_loss = result['bleu4'], result['loss']
        write_json(out/'initial_val_metrics.json', result)
        write_json(out/'best_val_predictions.json', preds)
        save(out/'best.pt', {'model': model.state_dict(), 'args': vars(args), 'epoch': 0})
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
            stop.fill_(int(args.patience > 0 and stale >= args.patience))
        if world > 1: dist.broadcast(stop, src=0)
        if stop.item(): break
    if world > 1: dist.destroy_process_group()


if __name__ == '__main__':
    main()
