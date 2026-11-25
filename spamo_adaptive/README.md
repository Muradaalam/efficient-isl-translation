# Adaptive SpaMo-OF reconstruction (iSign-10K)

This is a research reconstruction, not the authors' unpublished SpaMo-OF source. It imports the original SpaMo S2 wrapper, temporal CNN and vision projectors, and the original GMA network/checkpoint. The paper is https://aclanthology.org/2025.wslp-main.7/; upstream code: https://github.com/eddie-euijun-hwang/SpaMo and https://github.com/zacjiang/GMA.

## Kaggle setup

Use the existing iSign-10K dataset root with `manifest.csv` (`id,text,split,video_file`), and `videos/`. Keep official splits. Install only needed dependencies in the existing environment (do not install upstream historical torch pins): `python3 -m pip install einops scipy transformers peft accelerate sentencepiece sacrebleu rouge-score requests tqdm`. OpenCV and PyTorch are normally installed on Kaggle. The earlier PEFT/torchao conflict may require removing incompatible optional torchao. Enable internet and GPU T4 x2. Clone upstream repositories into `/kaggle/working/SpaMo` and `/kaggle/working/GMA`. GMA uses its official `checkpoints/gma-sintel.pth` demo checkpoint.

## Feature extraction

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u spamo_adaptive/extract_features.py --data /kaggle/working/isign_spamo_10k --output /kaggle/working/spamo_features/adaptive256 --spamo /kaggle/working/SpaMo --gma /kaggle/working/GMA --gma-checkpoint /kaggle/working/GMA/checkpoints/gma-sintel.pth --method adaptive --frames 256 --limit 2
```

Remove `--limit 2` for full extraction. For parallel extraction, pass `--shards 2 --shard 0` to GPU 0 and `--shards 2 --shard 1` to GPU 1, using the same output path. Atomic per-video feature files allow restarting. Check count is 10000; errors are not silently skipped. The same cache is reused for alignment and translation.

## Context reference bank

The paper uses Hindi, French and Spanish translations of training sentences. Supply a CSV `id,hi,fr,es` or use `prepare_context.py --data ... --output ...` to preview character count. Add `--execute` only if you have configured the billable Google Cloud Translation v2 API with a key in the Kaggle Secret `GOOGLE_TRANSLATE_KEY` and consent to sending training text to it. No videos are sent. For an explicitly different no-context ablation, pass `--no-context` consistently for align/train/test.

## Training

Use separate directories for each sampling method. XL is default; `--model-name google/flan-t5-large` is an optional deviation and must be identical for alignment, training and testing. Both stages use the same cached features. Alignment stage trains the upstream projection/TCN/MLP with frozen T5 embeddings; LoRA translation stage retains a contrastive auxiliary loss, as in released upstream. FP32 and non-reentrant gradient checkpointing avoid the earlier T4 FP16 issue. Both GPUs use DDP. To resume append `--resume` to the identical stage command; checkpoints are compact, but RNG state and exact alignment mid-epoch position are not restored. Preserve checkpoints and caches across Kaggle resets.

## Reconstruction decisions and limitations

* Adaptive spatial selection uses a 50/50 uniform + high grayscale-frame-difference budget; the RGB motion score replaces the older pose-hand score. For the GMA/VideoMAE branch, select original contiguous 16-flow-map windows from stride-8 candidates using the same hybrid ranking. Do not compute optical flow across skipped frames. The frame cap applies to spatial tokens; overlapping motion windows can cover more original frames. Short videos are not padded to the frame cap. GMA runs at square 256px, official Sintel checkpoint, 12 iterations. The paper does not specify all of those flow settings.
* VideoMAE features use the upstream first-token convention; CLIP uses the upstream S2 wrapper (scales 1,2). Cache float16, compute FP32; training uses FP32. Padding-aware contrastive pooling and integer TCN output lengths are correctness changes relative to upstream code. Short videos with insufficient joint tokens cause an explicit error.
* Translation instruction explicitly targets English (the paper's example says Indian). Context comes only from training references, excluding the current target text. Three languages in training; none during test, as in Table 1. The original exact multilingual reference texts are not assumed to be available.
* Alignment is a separate 1000-optimizer-step stage, then a new optimizer/schedule for translation; this is a reconstruction choice. Upstream initial LR 6e-4, cosine schedule, LoRA q/v rank 16 alpha 32 dropout .1. Maximum 20 translation epochs and patience 5 are project choices. Best checkpoint by corpus BLEU-4. DDP sampler can pad a sample and drop incomplete local batches. Validation and test use the full unshuffled split on one GPU. Beam sampling follows upstream (5 beams, top_p=.9); metrics can vary with generation RNG.
* Python syntax and sampling/window unit tests were verified locally. No real GPU feature extraction or training was executed in the authoring environment. Upstream model/checkpoint availability, current Kaggle package compatibility, GPU/host memory, runtime and translation quality still need validation. Do not claim exact reproduction or published performance.
