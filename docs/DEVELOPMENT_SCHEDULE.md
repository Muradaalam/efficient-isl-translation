# Development and Integration Schedule

**Project:** Efficient Continuous Indian Sign Language Translation  
**Team:** Murad, Shahjad, Rohit, Hamid  
**Schedule:** 12–18 October 2026  
**Goal:** Deliver a working, integrated baseline before starting advanced optimization experiments.

## 1. Daily Development Plan

| Date | Murad — Model | Shahjad — Features | Rohit — Sampling | Hamid — Evaluation |
|---|---|---|---|---|
| Oct 12 | GRU encoder | Extract pose from one MP4 | Uniform sampler | BLEU, WER, exact match |
| Oct 13 | Decoder and forward-pass test | Pose normalization and masks | Sampling tests | Inference interface |
| Oct 14 | **First integration checkpoint** | Verify pose compatibility | Verify sampler compatibility | Verify outputs |
| Oct 15 | Training loop | Batch pose extraction | Motion-aware sampling | Checkpoint loading |
| Oct 16 | Small training run | Motion features | Sampling comparisons | Evaluation pipeline |
| Oct 17 | Baseline training | Feature validation | Efficiency measurements | Latency benchmarks |
| Oct 18 | **Full integration checkpoint** | End-to-end feature testing | Sampling integration | Demo verification |

## 2. Mandatory Integration Checkpoints

### October 14 — First Compatibility Test

**Target:** One real MP4 passes through pose extraction, uniform sampling, and the model forward pass.

Acceptance criteria:
- Pose tensor and validity mask have agreed dimensions.
- Temporal sampler preserves frame order and masks.
- Model accepts sampled features.
- Basic automated tests pass.
- All members push their working implementations.

### October 16 — Training and Evaluation Test

**Target:** Train a small model, save a checkpoint, reload it, and evaluate predictions.

Acceptance criteria:
- Training completes on a small subset.
- Checkpoint can be saved and loaded.
- Inference produces text predictions.
- Evaluation metrics execute successfully.
- Experiment configuration is recorded.

### October 18 — Integrated Baseline

**Target:** Establish a reproducible end-to-end baseline.

Acceptance criteria:
- Video-to-English pipeline runs.
- Training and inference are reproducible.
- Shared dataset splits are used.
- Initial quality and efficiency metrics are recorded.
- Integration changes are merged through reviewed pull requests.

## 3. Daily Collaboration Rules

Each member must:
1. Work on their assigned feature branch.
2. Make meaningful commits as work progresses.
3. Push completed work regularly.
4. Report completed tasks, next tasks, and blockers.
5. Notify teammates before changing shared interfaces.
6. Add or update tests for new functionality.

## 4. Integration Rules

- Do not wait until the final day to combine modules.
- Integration issues take priority over advanced experiments.
- No untested code should be merged into `main`.
- Changes to landmark ordering, tensor shapes, tokenizer, or checkpoint format require coordination.
- All experiments must use the shared dataset manifest and fixed splits.
- Missed milestones must be rescheduled based on actual progress.

## 5. After October 18

Once the integrated baseline works, begin the research comparison:

- E1: Pose + uniform sampling.
- E2: Pose and motion + uniform sampling.
- E3: Pose + motion-aware sampling.
- E4: Pose and motion + motion-aware sampling.

Compare translation quality, inference latency, memory usage, and sequence reduction.

**Completion criterion:** A reproducible integrated baseline is required before final optimization results are reported.
