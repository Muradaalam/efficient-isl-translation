"""GloSL pose -> temporal CNN -> full fine-tuned FLAN-T5-Small.
Run from repository root: python3 -m scripts.train_flan_small ...
"""
import argparse
import json
import math
import os
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF
from jiwer import wer

from src.data.flan_dataset import FLANGloSLDataset

MODEL = 'google/flan-t5-small'


class PoseTranslator(nn.Module):
    def __init__(self, model_name=MODEL, landmarks=543):
        super().__init__()
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
        d = self.t5.config.d_model
        self.landmarks = landmarks
        # Position + displacement (each has x, y, z).
        self.spatial = nn.Sequential(nn.Linear(landmarks * 6, d), nn.LayerNorm(d), nn.GELU())
        self.temporal = nn.ModuleList([nn.Conv1d(d, d, 5, stride=2, padding=2) for _ in range(2)])
        self.norm = nn.LayerNorm(d)
        with torch.no_grad():
            norms = self.t5.get_input_embeddings().weight.float().norm(dim=-1)
            token_norm = norms.mean().item()
        self.initial_scale = token_norm / math.sqrt(d)
        # Log parameter ensures positive scale throughout training.
        self.log_scale = nn.Parameter(torch.tensor(math.log(max(self.initial_scale, 1e-6))))
        print(f'T5 hidden={d}, text norm={token_norm:.3f}, initial scale={self.initial_scale:.3f}', flush=True)

    def encode_pose(self, pose, valid_mask, frame_mask):
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
        x = self.spatial(torch.cat((p.flatten(2), delta.flatten(2)), dim=-1))
        x = x * mask.unsqueeze(-1)
        for conv in self.temporal:
            x = torch.nn.functional.gelu(conv(x.transpose(1, 2)).transpose(1, 2))
            # With kernel=5, padding=2, stride=2, output centers are 0,2,4,...
            # Existing sampler right-pads sequences. Downsample the frame mask likewise.
            mask = mask[:, ::2]
            x = x * mask.unsqueeze(-1)
        x = self.norm(x) * self.log_scale.exp()
        x = x * mask.unsqueeze(-1)
        return x, mask.long()

    def forward(self, pose, valid_mask, frame_mask, labels):
        x, mask = self.encode_pose(pose, valid_mask, frame_mask)
        return self.t5(inputs_embeds=x, attention_mask=mask, labels=labels, use_cache=False).loss

    @torch.no_grad()
    def translate(self, pose, valid_mask, frame_mask, beams=4, max_tokens=96):
        x, mask = self.encode_pose(pose, valid_mask, frame_mask)
        encoded = self.t5.get_encoder()(inputs_embeds=x, attention_mask=mask, return_dict=True)
        return self.t5.generate(encoder_outputs=encoded, attention_mask=mask, num_beams=beams,
                                max_new_tokens=max_tokens, do_sample=False, use_cache=False)


class Collator:
    def __init__(self, tokenizer, max_tokens):
        self.tokenizer, self.max_tokens = tokenizer, max_tokens

    def __call__(self, items):
        batch = {k: torch.stack([torch.as_tensor(item[k]) for item in items])
                 for k in ('pose', 'valid_mask', 'frame_mask')}
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
    for batch in loader:
        x = move(batch, device)
        if shuffled:
            if len(x['pose']) == 1:
                continue
            for key in ('pose', 'valid_mask', 'frame_mask'):
                x[key] = x[key].roll(1, dims=0)
        n = int((x['labels'] != -100).sum().item())
        losses += model(**x).item() * n
        tokens += n
        ids = model.translate(x['pose'], x['valid_mask'], x['frame_mask'], args.beams, args.max_tokens)
        preds.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
        refs.extend(batch['sentences'])
    result = metrics(preds, refs)
    result['loss'] = losses / max(tokens, 1)
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
        if not cache and not pred.strip():
            raise RuntimeError('Pretrained T5 generates blank text even without pose inputs; stop here.')


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
                            pose_column=args.pose_column, text_column=args.text_column)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--pose-root', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--pose-column', default='pose_file')
    ap.add_argument('--text-column', default='text')
    ap.add_argument('--method', default='adaptive', choices=['adaptive', 'uniform', 'motion-aware'])
    ap.add_argument('--frames', type=int, default=256)
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--mode', choices=['overfit', 'train', 'test'], default='train')
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
    ap.add_argument('--overfit-samples', type=int, default=32)
    ap.add_argument('--overfit-steps', type=int, default=200)
    ap.add_argument('--check-every', type=int, default=25)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--checkpoint', default=None)
    args = ap.parse_args()
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
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = PoseTranslator(args.model).to(device)
    if rank == 0 and args.mode != 'test':
        text_check(model, tokenizer, device)
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
        for key in ('model', 'method', 'frames'):
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
            embeds, mask = model.encode_pose(x['pose'], x['valid_mask'], x['frame_mask'])
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
        for key in ('model', 'method', 'frames', 'epochs', 'batch_size'):
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
        while steps < total:
            for batch in train_loader:
                loss = update(batch)
                steps += 1
                if steps % args.check_every == 0 or steps == total:
                    print(f'Overfit step {steps}/{total}, batch loss={loss:.4f}', flush=True)
                    result, preds = evaluate(model, val_loader, tokenizer, device, args)
                    history.append({'step': steps, **result})
                    write_json(out / 'overfit_history.json', history)
                    write_json(out / 'overfit_predictions.json', preds)
                    if result['exact_match'] >= .75:
                        print('Overfit target reached. This demonstrates train-set memorization, not generalization.', flush=True)
                        return
                if steps >= total: break
        print('Overfit run completed; inspect generated text and metrics before full training.', flush=True)
        return

    for epoch in range(start, args.epochs):
        if world > 1: sampler.set_epoch(epoch)
        stats = torch.zeros(2, dtype=torch.float64, device=device)
        bar = tqdm(train_loader, desc=f'Epoch {epoch+1}', disable=rank != 0)
        for batch in bar:
            loss = update(batch)
            n = int((batch['labels'] != -100).sum())
            stats += torch.tensor([loss*n, n], device=device, dtype=torch.float64)
            if rank == 0: bar.set_postfix(loss=f'{loss:.3f}')
        if world > 1: dist.all_reduce(stats)
        stop = torch.zeros(1, device=device, dtype=torch.int32)
        if rank == 0:
            result, preds = evaluate(model, val_loader, tokenizer, device, args)
            result.update(epoch=epoch+1, train_loss=(stats[0]/stats[1]).item(), scale=model.log_scale.exp().item())
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
