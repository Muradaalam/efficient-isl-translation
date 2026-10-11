# Team Development Rules
**Project:** Efficient Continuous Indian Sign Language Translation  
**Dataset:** ISL-CSLRT  
**Status:** Development contract v1.0

## 1. Project Objective

Build an efficient system that translates continuous Indian Sign Language videos into English sentences.

Research objectives:

1. Implement a reproducible pose-based translation baseline.
2. Evaluate whether motion features improve translation.
3. Evaluate whether motion-aware temporal sampling reduces computational cost.
4. Compare translation quality, inference speed, and memory consumption.
5. Integrate the best-performing configuration into a working demonstration.

The project must be evaluated as a research system, not merely as an application.

## 2. Team Responsibilities

### Murad — Model and Training
Owns:
- `src/models/`
- `src/training/`
- `configs/model.yaml`

Responsibilities:
- Implement the baseline sequence-to-sequence translator.
- Implement text tokenization and decoding.
- Implement training and validation loops.
- Support variable-length pose sequences.
- Implement checkpoint saving and loading.
- Compare baseline and optimized configurations.
- Maintain reproducible training settings.

Deliverables:
- Working baseline model.
- Training script.
- Saved model configuration.
- Training and validation logs.
- Model experiments.

### Shahjad — Pose and Motion Features
Owns:
- `src/features/`

Responsibilities:
- Extract pose landmarks from video using MediaPipe.
- Normalize landmark coordinates.
- Handle missing or low-confidence landmarks.
- Preserve frame ordering.
- Implement motion/velocity features.
- Produce reproducible feature files.
- Document landmark selection and feature dimensions.

Deliverables:
- Pose extraction pipeline.
- Normalization functions.
- Motion feature implementation.
- Feature documentation.
- Unit tests.

### Rohit — Temporal Sampling
Owns:
- `src/sampling/`

Responsibilities:
- Implement uniform frame sampling.
- Implement motion-aware sampling.
- Support configurable target sequence lengths.
- Preserve chronological order.
- Handle short and empty sequences.
- Measure sequence reduction.
- Compare sampling methods under identical conditions.

Deliverables:
- Uniform sampler.
- Motion-aware sampler.
- Sampling configurations.
- Unit tests.
- Compression and timing measurements.

### Hamid — Evaluation and Inference
Owns:
- `src/evaluation/`
- `src/inference/`
- `app/`

Responsibilities:
- Implement prediction and evaluation scripts.
- Calculate translation quality metrics.
- Measure inference latency and throughput.
- Measure model size and memory consumption.
- Evaluate saved checkpoints on fixed test splits.
- Develop the final demonstration.
- Prepare standardized experiment comparisons.

Deliverables:
- Evaluation pipeline.
- Inference pipeline.
- Efficiency benchmarks.
- Results tables.
- Working demonstration.

## 3. Dataset Rules

### 3.1 Source

Use the downloaded ISL-CSLRT corpus.

The source dataset contains:
- Sentence-level MP4 videos.
- Sentence-level extracted frames.
- Word-level extracted frames.
- Sentence annotation spreadsheet.
- Frame annotation spreadsheet.
- Word annotation spreadsheet.
- Sentence-to-gloss CSV.

Do not modify the original dataset files.

### 3.2 Dataset Location

Every member stores the dataset locally.

Use a configurable dataset root. Never hardcode absolute paths such as `/home/murad/...`.

All code must work when the dataset root changes.

### 3.3 Dataset Manifest

Create one shared manifest with these fields:

| Field | Meaning |
|---|---|
| `sample_id` | Unique stable sample identifier |
| `video_path` | Path relative to dataset root |
| `sentence` | English target sentence |
| `gloss` | Optional sign-gloss sequence |
| `signer_id` | Signer identity if reliably available |
| `split` | Train, validation, or test |
| `status` | Valid, missing, or unreadable |

The annotation spreadsheet is the authoritative source for sentence labels.

Do not infer English labels from video filenames.

Convert Windows-style paths to portable relative paths.

Preserve the original annotations.

### 3.4 Dataset Validation

Before full training:
- Match annotation paths against actual files.
- Identify missing and duplicate references.
- Verify that videos can be decoded.
- Record actual usable sample counts.
- Identify the number of unique sentences.
- Check whether reliable signer identifiers exist.
- Exclude invalid samples consistently.

The difference between 700 reported videos and 687 discovered MP4 files does not block development. Report the actual validated count.

### 3.5 Dataset Splits

One shared split file must be used by all members.

Rules:
- No sample may appear in multiple splits.
- Use a fixed random seed of `42`.
- Do not create separate random splits for individual experiments.
- Prevent duplicate or near-duplicate recordings from crossing splits.
- Do not tune model settings on the test set.

For the first baseline, use a fixed sentence-stratified recording split where feasible, approximately 70% training, 15% validation, and 15% testing.

This evaluates new recordings of mostly known sentences, **not translation of unseen sentences**.

A separate sentence-disjoint evaluation must be attempted to measure generalization to unseen sentences. Its limitations must be reported because the dataset contains a small sentence vocabulary.

If reliable signer IDs are recovered, a signer-disjoint evaluation may be added.

Final split assignments must be generated only after validating the manifest.

## 4. Shared Data Interface

### 4.1 Dataset Sample

Each sample must provide:

```python
{
    "sample_id": str,
    "pose": Tensor,       # [T, J, 3]
    "valid_mask": Tensor, # [T, J]
    "text": str,
    "gloss": str | None
}
```

`T` = frames  
`J` = selected landmarks  
`3` = normalized x, y, z coordinates

`valid_mask` identifies landmarks with usable detections.

Landmark count, order, coordinate normalization, and confidence thresholds must be fixed in a shared feature configuration before full extraction.

### 4.2 Feature Representation

Use:
- Selected body landmarks.
- Left-hand landmarks.
- Right-hand landmarks.

Facial landmarks are optional and excluded from the initial baseline unless justified experimentally.

All feature sequences must preserve the same landmark order.

Missing detections must not silently be interpreted as genuine zero-coordinate measurements.

### 4.3 Motion Features

Define motion as temporal differences between consecutive pose frames.

Motion features must:
- Use the same landmark ordering.
- Handle the first frame explicitly.
- Handle missing detections.
- Preserve the temporal dimension.
- Document whether time differences are normalized by frame interval.

Pose-only and pose-plus-motion configurations must remain independently selectable.

### 4.4 Temporal Sampling

The sampler must accept:

```python
sample(
    pose,
    valid_mask,
    target_frames,
    method
)
```

It must return:
- Sampled pose sequence.
- Corresponding sampled validity mask.
- Original selected frame indices.

Rules:
- Preserve chronological ordering.
- Never reorder landmark channels.
- Never use target sentence labels to choose frames.
- Apply the same sampling implementation during comparable experiments.
- Record the selected frame indices for debugging.

### 4.5 Model Interface

The translation model must accept batched feature sequences with valid sequence lengths or attention masks.

Required behavior:
- Handle variable-length inputs.
- Support training and inference modes.
- Return token predictions or token logits.
- Support deterministic evaluation.
- Allow checkpoints to be loaded without rewriting the architecture.

Use one shared English tokenizer and vocabulary created from training data only.

Do not build separate tokenizers for each experiment.

## 5. Baseline Model Rules

The first model must be lightweight enough for available cloud GPU resources.

Recommended initial architecture:
- Pose feature projection.
- GRU-based temporal encoder.
- Attention-based autoregressive decoder.
- English token vocabulary.

A Transformer-based alternative can be tested after the baseline works.

Do not begin with a large pretrained language model.

The first milestone is a complete working pipeline, not maximum translation accuracy.

## 6. Evaluation Rules

### Translation Quality

Report:
- BLEU-1 through BLEU-4.
- Word error rate where appropriate.
- Exact sentence-match accuracy.
- Representative reference/prediction examples.

Use consistent text normalization and metric implementations.

Exact-match accuracy is especially useful for the dataset's limited sentence vocabulary, but it must not be presented as evidence of open-vocabulary translation.

### Efficiency

Report:
- Average inference latency per sample.
- Throughput in samples per second.
- Input sequence length.
- Parameter count.
- Model checkpoint size.
- Peak memory consumption where measurable.
- Hardware and batch size.

Separate pose extraction time from model inference time.

### Experiment Comparisons

At minimum, compare:

| ID | Features | Sampling |
|---|---|---|
| E1 | Pose | Uniform |
| E2 | Pose + motion | Uniform |
| E3 | Pose | Motion-aware |
| E4 | Pose + motion | Motion-aware |

Keep training settings, dataset splits, and model capacity consistent wherever possible.

Document any differences.

## 7. Reproducibility Rules

Every experiment must record:
- Experiment ID.
- Git commit hash.
- Dataset split version.
- Feature configuration.
- Sampling configuration.
- Model configuration.
- Random seed.
- Training epochs and batch size.
- Hardware.
- Translation metrics.
- Efficiency metrics.

Store lightweight reports in Git.

Do not commit large checkpoints, extracted features, or original dataset files.

## 8. GitHub Collaboration

The stable integration branch is `main`.

Feature branches:

- Murad: `feature/model-training`
- Shahjad: `feature/pose-motion`
- Rohit: `feature/temporal-sampling`
- Hamid: `feature/evaluation-demo`

Rules:
1. Each member commits their own work.
2. Do not develop directly on `main`.
3. Open pull requests for integration.
4. Require at least one teammate's review.
5. Run relevant tests before merging.
6. Resolve merge conflicts before merging.
7. Coordinate changes to shared interfaces.
8. Do not commit datasets, checkpoints, credentials, or generated outputs.
9. Commit small, meaningful units of work.
10. Never create artificial commits merely to increase the commit count.

Suggested commit prefixes:
- `feat:` New functionality.
- `fix:` Bug fix.
- `test:` Tests.
- `docs:` Documentation.
- `refactor:` Internal restructuring.
- `chore:` Configuration or maintenance.
- `exp:` Experiment-related changes.

## 9. Shared File Ownership

Shared files:
- `src/data/dataset.py`
- `configs/dataset.yaml`
- `requirements.txt`
- Dataset manifests and split definitions.

Murad coordinates integration of these files, but changes require review from affected module owners.

No member may silently change:
- Landmark ordering.
- Feature dimensions.
- Tokenizer format.
- Split assignments.
- Dataset sample schema.
- Checkpoint format.
- Evaluation definitions.

Any such change requires a reviewed pull request.

## 10. Testing and Integration

Each module must include tests for its public interface.

Required tests:
- Pose extraction handles a valid sample video.
- Missing landmarks are handled consistently.
- Motion features preserve expected dimensions.
- Temporal sampling preserves frame order.
- Model accepts variable-length batches.
- Model can perform a forward and backward pass.
- Inference loads a saved checkpoint.
- Evaluation produces deterministic results on fixed predictions.

Before full training, the integrated pipeline must pass a small end-to-end smoke test.

## 11. Integration Order

**Milestone 1 — Dataset foundation**
- Validate annotations and video paths.
- Create manifest.
- Create fixed splits.
- Finalize pose representation.

**Milestone 2 — Independent modules**
- Shahjad implements pose and motion features.
- Rohit implements temporal sampling.
- Murad implements the baseline model and training.
- Hamid implements evaluation and inference.

All members use the same documented interfaces.

**Milestone 3 — Baseline integration**
- Connect modules.
- Run smoke tests.
- Train the first baseline.
- Validate inference and evaluation.

**Milestone 4 — Research experiments**
- Compare pose and motion features.
- Compare temporal sampling strategies.
- Measure translation quality and efficiency.

**Milestone 5 — Final delivery**
- Integrate the best supported configuration.
- Prepare the demonstration.
- Produce comparison tables.
- Document limitations.
- Prepare the research report and presentation.

## 12. Definition of Done

A module is complete only when:
- Its implementation works.
- Its input/output interface is documented.
- Relevant tests pass.
- Another member can import and use it.
- Its dependencies are documented.
- It is committed and reviewed.
- It integrates without breaking the shared pipeline.

The project is complete only when a video can pass through preprocessing, temporal sampling, translation, and evaluation using a reproducible configuration.

## 13. Known Dataset Limitations

The team must acknowledge:
- The available dataset is relatively small.
- Many recordings share the same target sentences.
- Pre-extracted pose features were not found.
- Reliable signer IDs have not yet been verified.
- Some annotation paths may need normalization.
- Generalization to unseen sentences may be limited.

Research conclusions must not exceed what the experiments demonstrate.

### Dataset Contract v1.0 — Final Split Policy

- The baseline uses 492 validated spreadsheet-annotated sentence-video records.
- The 195 unreferenced MP4 files are excluded pending separate content verification.
- The canonical split file is `experiments/isl_manifest_split.csv`.
- The split generator uses random seed 42.
- All experiments must use exactly the same sample IDs and split assignments.
- The baseline split is sentence-stratified at the recording level.
- Single-recording sentences belong to the training set only.
- This evaluation measures performance on new recordings of largely known sentences.
- Signer-independent and unseen-sentence generalization are not established by this split.
- Any revised split must receive a new version and be evaluated separately.
- Before reporting final results, check for duplicate video content and possible signer leakage.