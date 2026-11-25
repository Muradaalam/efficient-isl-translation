"""Evaluate a trained V2 checkpoint without updating weights.
All retrieval candidates come from TRAIN. Validation targets are used only for scoring.
Run from repository root: python3 -m scripts.diagnose_flan_v2 ...
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from tqdm import tqdm
from scripts.train_flan_v2 import PoseTranslator, Collator, make_dataset, metrics, move


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--pose-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--probe-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    if a.batch_size < 2 or a.probe_size < 2:
        raise ValueError('batch and probe size must be at least 2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    args = SimpleNamespace(**state['args'])
    args.manifest, args.pose_root = a.manifest, a.pose_root
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = PoseTranslator(args.model, landmarks=args.landmarks,
                           contrastive_weight=args.contrastive_weight,
                           prefix_weight=args.prefix_weight,
                           prefix_tokens=args.prefix_tokens)
    model.load_state_dict(state['model'], strict=True)
    del state
    model.to(device).eval()
    collator = Collator(tokenizer, args.max_tokens)
    train, val = make_dataset(args, 'train'), make_dataset(args, 'val')

    def loader(ds):
        return DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                          collate_fn=collator, num_workers=a.workers,
                          pin_memory=device.type == 'cuda')

    def pose_encode(x):
        em, mask = model.encode_pose(x['pose'], x['valid_mask'], x['frame_mask'], x['indices'])
        enc = model.t5.get_encoder()(inputs_embeds=em, attention_mask=mask, return_dict=True)
        return enc, mask

    @torch.inference_mode()
    def bank(ds, name, include_text):
        poses, texts, sentences = [], [], []
        for batch in tqdm(loader(ds), desc=name):
            x = move(batch, device)
            enc, mask = pose_encode(x)
            poses.append(model.pooled(enc.last_hidden_state, mask).float().cpu())
            if include_text:
                tm = x['labels'].ne(-100)
                ids = x['labels'].masked_fill(~tm, model.t5.config.pad_token_id)
                h = model.t5.get_encoder()(input_ids=ids, attention_mask=tm.long(), return_dict=True)
                texts.append(model.pooled(h.last_hidden_state, tm).float().cpu())
            sentences.extend(batch['sentences'])
        return torch.cat(poses), torch.cat(texts) if include_text else None, sentences

    @torch.inference_mode()
    def probe(ds, name, seed):
        g = torch.Generator().manual_seed(seed)
        count = min(a.probe_size, len(ds))
        ix = torch.randperm(len(ds), generator=g)[:count].tolist()
        if count % a.batch_size == 1: ix = ix[:-1]
        correct, shuffled, tokens = 0., 0., 0
        preds, refs = [], []
        for batch in tqdm(loader(Subset(ds, ix)), desc=name):
            x = move(batch, device)
            n = int(x['labels'].ne(-100).sum())
            enc, mask = pose_encode(x)
            ce = model.t5(encoder_outputs=enc, attention_mask=mask, labels=x['labels'], use_cache=False).loss
            y = {**x}
            for key in ('pose','valid_mask','frame_mask','indices'):
                y[key] = x[key].roll(1, 0)
            enc2, mask2 = pose_encode(y)
            bad = model.t5(encoder_outputs=enc2, attention_mask=mask2, labels=x['labels'], use_cache=False).loss
            ids = model.t5.generate(encoder_outputs=enc, attention_mask=mask, num_beams=4,
                                    max_new_tokens=min(args.max_tokens, 96), do_sample=False, use_cache=False)
            correct += ce.item()*n
            shuffled += bad.item()*n
            tokens += n
            preds.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
            refs.extend(batch['sentences'])
        result = dict(metrics(preds, refs), correct_ce=correct/tokens,
                      shuffled_ce=shuffled/tokens, count=len(refs))
        result['grounding_gap'] = result['shuffled_ce'] - result['correct_ce']
        return result, [dict(reference=r, prediction=p) for r,p in zip(refs,preds)]

    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, ds in [('train_probe',train),('val_probe',val)]:
        result[name], preds = probe(ds, name, 42)
        print(name, json.dumps(result[name]), flush=True)
        (out / f'{name}_predictions.json').write_text(json.dumps(preds, indent=2, ensure_ascii=False))
    (out / 'diagnostic.json').write_text(json.dumps(result, indent=2))

    # Retrieval: compare every validation pose against TRAIN-only candidate vectors.
    train_pose, train_text, train_sentences = bank(train, 'TRAIN candidate bank', True)
    val_pose, _, val_sentences = bank(val, 'Validation queries', False)
    for name, vectors in [('pose_to_pose', train_pose), ('pose_to_text', train_text)]:
        nearest = (val_pose @ vectors.T).argmax(dim=1).tolist()
        predictions = [train_sentences[i] for i in nearest]
        result[name] = metrics(predictions, val_sentences)
        print(name, json.dumps(result[name]), flush=True)
        (out / f'{name}_predictions.json').write_text(json.dumps([
            dict(reference=r, prediction=p) for r,p in zip(val_sentences,predictions)
        ], indent=2, ensure_ascii=False))
    (out / 'diagnostic.json').write_text(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
