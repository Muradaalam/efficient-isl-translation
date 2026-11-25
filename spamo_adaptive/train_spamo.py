"""Two-stage SpaMo-OF reconstruction on cached features (torchrun compatible)."""
import argparse
import csv
import json
import math
import os
import random
from datetime import timedelta
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from extract_features import load_module, key, read_manifest


class Features(Dataset):
    def __init__(self, rows, root):
        self.rows, self.root = rows, Path(root)
        missing = [r['id'] for r in rows if not (self.root/f'{key(r)}.npz').is_file()]
        if missing: raise ValueError(f'{len(missing)} feature files missing: {missing[:3]}')
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        row = self.rows[i]
        with np.load(self.root/f'{key(row)}.npz') as data:
            s = torch.from_numpy(data['spatial'].astype(np.float32))
            m = torch.from_numpy(data['motion'].astype(np.float32))
        return s,m,row['text'].strip().lower(),row['id']


def collate(batch): return batch


def mask_from_lengths(lengths, width, device):
    return torch.arange(width,device=device)[None,:] < torch.tensor(lengths,device=device)[:,None]


def mean_masked(x, mask):
    return (x*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)


class Model(nn.Module):
    def __init__(self, upstream, model_name, stage):
        super().__init__()
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from peft import LoraConfig, get_peft_model, TaskType
        tconv = load_module('upstream_tconv',Path(upstream)/'spamo/tconv.py')
        proj = load_module('upstream_projector',Path(upstream)/'spamo/mm_projector.py')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Stage 1 only needs frozen token embeddings, not a language-model forward.
        base = AutoModelForSeq2SeqLM.from_pretrained(model_name,torch_dtype=torch.float32)
        d = base.config.d_model
        self.spatio_proj = proj.build_vision_projector('linear',2048,768)
        self.spatiotemp_proj = proj.build_vision_projector('linear',1024,768)
        self.temporal_encoder = tconv.TemporalConv(768,768)
        self.fusion_proj = proj.build_vision_projector('mlp2x_gelu',768,d)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))
        self.stage = stage
        if stage=='align':
            self.text_embed = base.get_input_embeddings()
            self.text_embed.requires_grad_(False)
            del base
        else:
            self.t5 = get_peft_model(base,LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,r=16,lora_alpha=32,lora_dropout=.1,target_modules=['q','v'],bias='none'))
            self.t5.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            self.t5.config.use_cache=False

    def embed(self, ids):
        return self.text_embed(ids) if self.stage=='align' else self.t5.get_input_embeddings()(ids)

    def encode(self, batch):
        device=self.logit_scale.device
        seq=[]
        for s,m,_,_ in batch:
            seq.append(torch.cat([self.spatio_proj(s.to(device)),self.spatiotemp_proj(m.to(device))],dim=0))
        lengths=[len(x) for x in seq]
        x=pad_sequence(seq,batch_first=True).transpose(1,2)
        h=self.temporal_encoder(x,torch.tensor(lengths,device=device))['visual_feat'].transpose(0,1)
        # Integer floor at each pooling step, matching actual convolution output.
        lengths=[((n-4)//2-4)//2 for n in lengths]
        mask=mask_from_lengths(lengths,h.shape[1],device)
        return self.fusion_proj(h),mask

    def tokenize(self, texts, limit):
        return self.tokenizer(texts,padding=True,truncation=True,max_length=limit,return_tensors='pt').to(self.logit_scale.device)

    def contrastive(self, z, mask, target):
        with torch.no_grad():
            y=mean_masked(self.embed(target.input_ids),target.attention_mask)
        v=F.normalize(mean_masked(z,mask),dim=-1)
        y=F.normalize(y,dim=-1)
        # Gather across GPUs with gradients for visual embeddings. Equal local batches required.
        if dist.is_initialized() and self.training:
            from torch.distributed.nn.functional import all_gather
            v=torch.cat(all_gather(v),dim=0)
            y=torch.cat(all_gather(y),dim=0)
        logits=(v@y.T)*self.logit_scale.clamp(max=math.log(100)).exp()
        labels=torch.arange(len(v),device=v.device)
        return (F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))/2

    def prompted(self,z,mask,prompts):
        p=self.tokenize(prompts,256)
        pe=self.embed(p.input_ids)
        seq=[torch.cat([v[vm],t[tm.bool()]],dim=0) for v,vm,t,tm in zip(z,mask,pe,p.attention_mask)]
        x=pad_sequence(seq,batch_first=True)
        return x,mask_from_lengths([len(v) for v in seq],x.shape[1],x.device).long()

    def forward(self,batch,prompts=None):
        z,mask=self.encode(batch)
        target=self.tokenize([b[2] for b in batch],64)
        c=self.contrastive(z,mask,target)
        if self.stage=='align': return c,c.detach(),c.detach()
        x,attn=self.prompted(z,mask,prompts)
        labels=target.input_ids.masked_fill(target.attention_mask==0,-100)
        t=self.t5(inputs_embeds=x,attention_mask=attn,decoder_attention_mask=target.attention_mask,labels=labels,use_cache=False).loss
        return t+c,t.detach(),c.detach()

    @torch.no_grad()
    def generate(self,batch,prompts):
        z,mask=self.encode(batch)
        x,attn=self.prompted(z,mask,prompts)
        # Same beam width and sampling settings as released SpaMo code.
        ids=self.t5.generate(inputs_embeds=x,attention_mask=attn,num_beams=5,max_length=64,top_p=.9,do_sample=True,use_cache=True)
        return [s.lower().strip() for s in self.tokenizer.batch_decode(ids,skip_special_tokens=True)]


class Prompts:
    def __init__(self,rows,context_csv,no_context):
        self.examples=[]
        self.no_context=no_context
        if no_context: return
        if not context_csv: raise ValueError('Provide --context-csv with id,hi,fr,es or explicitly choose --no-context (ablation)')
        with open(context_csv,newline='',encoding='utf-8') as f: translations={r['id']:r for r in csv.DictReader(f)}
        for r in rows:
            if r['id'] in translations:
                tr=translations[r['id']]
                if all(tr.get(k,'').strip() for k in ('hi','fr','es')):
                    self.examples.append((r['text'].strip().lower(),tr))
        if len({x[0] for x in self.examples})<2: raise ValueError('Not enough distinct multilingual training references')
    def __call__(self,batch,train):
        base='Translate the given sentence into English.'
        # Table 1 evaluates without in-context examples; no validation/test target exposure.
        if self.no_context or not train: return [base]*len(batch)
        result=[]
        for _,_,target,_ in batch:
            ref,tr=random.choice(self.examples)
            while ref==target:
                ref,tr=random.choice(self.examples)
            result.append(base+' '+' '.join(f'{tr[k]} = {ref}' for k in ('hi','fr','es')))
        return result


def compact_state(model,alignment_only=False):
    prefixes=('spatio_proj.','spatiotemp_proj.','temporal_encoder.','fusion_proj.')
    return {k:v.detach().cpu() for k,v in model.state_dict().items() if k.startswith(prefixes) or k=='logit_scale' or (not alignment_only and 'lora_' in k)}


def restore(model,state):
    expected=set(compact_state(model,alignment_only=not any('lora_' in k for k in state)))
    if set(state)!=expected: raise ValueError('Checkpoint parameter names do not match architecture')
    model.load_state_dict(state,strict=False)


def save(path,state):
    tmp=path.with_suffix('.tmp'); torch.save(state,tmp); tmp.replace(path)


def validate(model,loader,prompts,generate=True):
    from sacrebleu.metrics import BLEU
    from rouge_score import rouge_scorer
    model.eval(); losses=[]; hyp=[]; ref=[]
    with torch.no_grad():
        for batch in loader:
            p=prompts(batch,False)
            _,t,_=model(batch,p); losses.append(float(t))
            if generate:
                hyp.extend(model.generate(batch,p)); ref.extend([b[2] for b in batch])
    result={'val_loss':float(np.mean(losses))}
    if generate:
        for n in range(1,5): result[f'bleu{n}']=BLEU(max_ngram_order=n).corpus_score(hyp,[ref]).score
        scorer=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=False)
        result['rougeL']=100*float(np.mean([scorer.score(r,h)['rougeL'].fmeasure for h,r in zip(hyp,ref)]))
        result['blank_pct']=100*sum(not h for h in hyp)/len(hyp)
        for h,r in list(zip(hyp,ref))[:5]: print('Reference:',r,'\nPredicted:',repr(h),flush=True)
    return result


def main():
    from tqdm import tqdm
    from transformers import get_cosine_schedule_with_warmup
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True); p.add_argument('--features',required=True); p.add_argument('--spamo',required=True); p.add_argument('--output',required=True)
    p.add_argument('--stage',choices=['align','train','test'],required=True); p.add_argument('--model-name',default='google/flan-t5-xl')
    p.add_argument('--context-csv'); p.add_argument('--no-context',action='store_true'); p.add_argument('--resume',action='store_true')
    p.add_argument('--batch-size',type=int,default=1); p.add_argument('--accum',type=int,default=8); p.add_argument('--workers',type=int,default=2)
    p.add_argument('--epochs',type=int,default=20); p.add_argument('--patience',type=int,default=5); p.add_argument('--align-steps',type=int,default=1000); p.add_argument('--lr',type=float,default=6e-4); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--limit',type=int,default=0,help='Optional per-split limit for integration test; use separate output')
    a=p.parse_args()
    local=int(os.environ.get('LOCAL_RANK',0)); torch.cuda.set_device(local); device=torch.device('cuda',local)
    world=int(os.environ.get('WORLD_SIZE',1)); rank=int(os.environ.get('RANK',0))
    if world>1: dist.init_process_group('nccl',timeout=timedelta(hours=2))
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    root=Path(a.output); root.mkdir(parents=True,exist_ok=True)
    settings=json.loads((Path(a.features)/'settings.json').read_text())
    run_config={'model':a.model_name,'feature_settings':settings,'context':not a.no_context}
    config=root/'run.json'
    if config.exists() and json.loads(config.read_text())!=run_config: raise ValueError('Run settings mismatch')
    if rank==0: config.write_text(json.dumps(run_config,indent=2))
    rows=read_manifest(a.data); train_rows=[r for r in rows if r['split']=='train']; val_rows=[r for r in rows if r['split']=='val']; test_rows=[r for r in rows if r['split']=='test']
    if a.limit: train_rows,val_rows,test_rows=[r[:a.limit] for r in (train_rows,val_rows,test_rows)]
    stage='align' if a.stage=='align' else 'train'
    model=Model(a.spamo,a.model_name,stage).to(device)
    if stage=='train' and a.stage!='test' and not a.resume:
        restore(model,torch.load(root/'alignment.pt',map_location='cpu',weights_only=True)['model'])
    prompts=Prompts(train_rows,a.context_csv,a.no_context) if stage=='train' and a.stage!='test' else Prompts(train_rows,None,True)
    def loader(rows,batch_size,sampler=None,drop=False):
        return DataLoader(Features(rows,a.features),batch_size=batch_size,sampler=sampler,shuffle=sampler is None and drop,drop_last=drop,collate_fn=collate,num_workers=a.workers,pin_memory=True)
    if a.stage=='test':
        if world!=1: raise ValueError('Run test with one GPU')
        restore(model,torch.load(root/'best.pt',map_location='cpu',weights_only=True)['model'])
        result=validate(model,loader(test_rows,a.batch_size),prompts)
        print(result); (root/'test_metrics.json').write_text(json.dumps(result,indent=2)); return
    train=Features(train_rows,a.features)
    sampler=DistributedSampler(train,num_replicas=world,rank=rank,shuffle=True,seed=a.seed,drop_last=False)
    train_loader=DataLoader(train,batch_size=a.batch_size,sampler=sampler,drop_last=True,collate_fn=collate,num_workers=a.workers,pin_memory=True)
    if not len(train_loader): raise ValueError('Empty training loader')
    if a.stage=='align' and a.batch_size*world<2: raise ValueError('Contrastive alignment needs at least two examples per global batch')
    val_loader=loader(val_rows,a.batch_size) if rank==0 and a.stage=='train' else None
    wrapped=DDP(model,device_ids=[local],broadcast_buffers=True) if world>1 else model
    params=[v for v in model.parameters() if v.requires_grad]
    opt=torch.optim.AdamW(params,lr=a.lr,betas=(.9,.98),eps=1e-8,weight_decay=.01)
    steps_per_epoch=math.ceil(len(train_loader)/a.accum)
    total=a.align_steps if stage=='align' else steps_per_epoch*a.epochs
    scheduler=get_cosine_schedule_with_warmup(opt,max(1,int(total*.1)),total)
    epoch,step,best,stale=0,0,-1.,0
    checkpoint=root/('alignment.pt' if stage=='align' else 'latest.pt')
    if a.resume:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True); restore(model,state['model']); opt.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler']); epoch,step,best,stale=(state[k] for k in ('epoch','step','best','stale'))
        # Alignment checkpoints can be mid-epoch; resume shuffling on next epoch, not exact batch.
    def state_dict(): return {'model':compact_state(model,stage=='align'),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'step':step,'best':best,'stale':stale}
    if rank==0: print(f'{stage=} {world=} {total=} {len(train)=} trainable={sum(v.numel() for v in params):,}',flush=True)
    while step<total and (stage=='align' or epoch<a.epochs):
        epoch+=1; sampler.set_epoch(epoch); wrapped.train(); opt.zero_grad(set_to_none=True)
        bar=tqdm(train_loader,desc=f'{stage} epoch {epoch}',disable=rank!=0)
        for i,batch in enumerate(bar,1):
            update=i%a.accum==0 or i==len(train_loader)
            group=min(a.accum,len(train_loader)-((i-1)//a.accum)*a.accum)
            with wrapped.no_sync() if world>1 and not update else nullcontext():
                loss,t,c=wrapped(batch,prompts(batch,True) if stage=='train' else None)
                valid=torch.isfinite(loss).int()
                if world>1: dist.all_reduce(valid,op=dist.ReduceOp.MIN)
                if not valid.item(): raise RuntimeError('Nonfinite loss')
                (loss/group).backward()
            if update:
                torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); scheduler.step(); opt.zero_grad(set_to_none=True); step+=1
                if rank==0 and stage=='align' and (step%100==0 or step==total): save(checkpoint,state_dict())
                if step>=total: break
            if rank==0 and i%20==0: bar.set_postfix(loss=f'{loss.item():.3f}',contrastive=f'{c.item():.3f}',step=step)
        if stage=='align':
            if rank==0: save(checkpoint,state_dict())
            if world>1: dist.barrier()
            continue
        stop=torch.zeros(1,device=device)
        if rank==0:
            result=validate(model,val_loader,prompts); result['epoch']=epoch; print(result,flush=True)
            with (root/'history.jsonl').open('a') as f: f.write(json.dumps(result)+'\n')
            if result['bleu4']>best:
                best=result['bleu4']; stale=0; save(root/'best.pt',{'model':compact_state(model),'epoch':epoch,'metrics':result})
            else: stale+=1
            save(checkpoint,state_dict()); print(f'Checkpoint saved; patience {stale}/{a.patience}',flush=True)
            if stale>=a.patience: stop.fill_(1)
        if world>1: dist.broadcast(stop,src=0)
        if stop.item(): break
    if world>1: dist.destroy_process_group()

if __name__=='__main__': main()
