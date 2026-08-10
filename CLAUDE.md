# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A PyTorch research project implementing a "Viseme-Phoneme Adaptive Model" for lip reading on the GRID
corpus. The model reads mouth-crop video frames, predicts visemes (visually distinguishable lip shapes),
then disambiguates them into phonemes using a learned confusion-matrix decoder and an adaptive fusion of
visual features with a contextual language model. There is no test suite, linter, or build system
configured — this is an experimental/research codebase, not a packaged library.

## Commands

There is no packaging, linting, or automated test setup. Everything is run directly via `python`.

```bash
# Install dependencies
pip install -r requirements.txt
# dlib is also required (see dlib-19.24.1-cp311-cp311-win_amd64.whl for a Windows/py3.11 prebuilt wheel)

# Train (mode is positional; subjects/model-type/etc. are flags)
python main.py train --model-type full --train-subjects s1 --val-subjects s2 --test-subjects s3

# Evaluate a checkpoint
python main.py evaluate --checkpoint-path checkpoints/best_model.pth --model-type full

# Run the inference demo (prints per-sample viseme/phoneme predictions)
python main.py inference --checkpoint-path checkpoints/best_model.pth

# Print dataset statistics (frame/sequence shapes, viseme distribution, etc.)
python main.py analyze

# Generate placeholder sample data/directories for smoke-testing the pipeline
python main.py create_sample

# Standalone manual test of a single video against a checkpoint
python test.py <model_path> <video_path>
```

`main_old.py` is a stale duplicate of `main.py` — prefer `main.py`. `src/trainer.py` also has its own
`main()` for a minimal training run; `main.py train` is the primary entry point and supersedes it.

## Data layout

The `GRIDDataset` (`src/data_loader.py`) expects, per subject (e.g. `s1`, `s2`, `s3`):

```
data/videos/<subject>/<utterance_id>.mp4|.mpg
data/alignments/<subject>/<utterance_id>.align   # "start end label" lines, GRID-style forced alignments
```

None of this data is checked into the repo — `python main.py create_sample` only creates placeholder
`.align` files and empty video stubs for pipeline smoke-testing, not real videos.

Dataset loading requires `shape_predictor_68_face_landmarks.dat` (dlib's 68-point facial landmark model,
present at the repo root) to crop the mouth ROI from each frame via `dlib.get_frontal_face_detector()`.

## Architecture

Data flows through five stages, wired together in `VisemePhonemeAdaptiveModel`
(`models/fusion_network.py`), the single top-level `nn.Module`:

1. **Visual encoding** (`models/visual_encoder.py`) — `EnhancedVisualEncoder` = `LipROI` (a 2D
   spatial-attention gate applied per-frame) → `VisualEncoder` (4-block 3D CNN over the frame sequence,
   flattened and projected to `Config.VISUAL_FEATURE_DIM`-dim per-frame features). Note `VisualEncoder`
   builds `fc1`/`fc2` from a dummy forward pass at init but then lazily rebuilds `fc1_dynamic`/
   `fc2_dynamic` on first real forward call if the flattened conv size differs — the static `fc1`/`fc2`
   layers end up unused dead weights.

2. **Sequence modeling** (`models/sequence_model.py`) — `HybridSequenceModel` runs an `LSTMSequenceModel`
   over the visual features, concatenates the LSTM hidden state back with the original features, then
   feeds that through a `TransformerSequenceModel` (with sinusoidal positional encoding) to produce the
   final per-frame hidden representation (`Config.HIDDEN_DIM`-dim) and viseme logits.

3. **Language modeling** (`src/language_model.py`) — only invoked when `model_type='full'`.
   `ContextualLanguageModel` wraps `HierarchicalLanguageModel` (a `GRIDLanguageModel` word LSTM, whose
   position embeddings encode the fixed GRID sentence grammar `[command][color][preposition][letter]
   [digit][adverb]`, cross-attended with a `PhonemeLanguageModel` transformer) and adapts it with a
   per-speaker embedding (`speaker_ids` ∈ {s1, s2, s3} → {0, 1, 2}, hardcoded in `main.py`/`src/trainer.py`).

4. **Adaptive fusion** (`src/adaptive_fusion.py`) — `MultiScaleAdaptiveFusion` runs the same
   `AdaptiveFusionModule` at 4 temporal pooling scales (1/3/5/7 frames), each gating visual vs. language
   features per-timestep based on the timestep's ambiguity score (via a small `ambiguity_gate` MLP), then
   combines the 4 scale outputs with a learned scale-attention. Ambiguity scores come from
   `viseme_mapper.get_ambiguity_score` (higher when a viseme maps to more phonemes) and are computed
   automatically from predicted visemes if not supplied.

5. **Viseme→phoneme decoding** (`src/adaptive_fusion.py`) — `ConfusionMatrixDecoder` starts from the
   static phoneme→viseme confusion matrix (`src/viseme_mapping.py`, 12 visemes total, indices 0–11), adds
   a learnable adjustment, converts fused viseme probabilities to phoneme probabilities via matrix
   multiply, then refines them with a small phoneme-sequence BiLSTM.

`src/viseme_mapping.py` (`viseme_mapper`, a module-level singleton) is the shared source of truth for the
phoneme↔viseme mapping and ambiguity scores; it's imported by the data loader, the fusion model, and the
adaptive fusion decoder alike — treat it as canonical rather than re-deriving mappings elsewhere.

`ModelFactory` (`models/fusion_network.py`) builds three model variants from the same class:
`baseline` (adaptive fusion replaced with `nn.Identity`, so `fused_output` skips fusion but still expects
`Config.HIDDEN_DIM`-shaped input — this variant is currently broken since `AdaptiveFusionModule`'s output
projects to `num_visemes`, not `HIDDEN_DIM`), `visual_only` (language loss weight zeroed but the language
model still runs), and `full` (default).

`src/trainer.py`'s `LipReadingTrainer` handles the training loop, TensorBoard logging (`runs/`),
checkpointing (`checkpoints/`, gradient clipping at norm 1.0, `ReduceLROnPlateau`, early stopping via
`Config.PATIENCE`), and a final test-set evaluation broken down by high/low ambiguity
(`Config.AMBIGUITY_THRESHOLD`). `src/evaluator.py`'s `LipReadingEvaluator` is the more comprehensive
standalone evaluation path used by `main.py evaluate`, producing confusion matrix plots (`plots/`) and a
text report (`results/`).

All tunable constants (paths, model dims, video/frame params, training hyperparameters, the GRID
vocabulary/phoneme set) live in one place: `src/config.py`'s `Config` class (accessed as class attributes,
not instantiated). `Config.DEVICE` is mutated at runtime by `main.py` based on `--device`/CUDA
availability — code elsewhere reads `Config.DEVICE` live rather than capturing it at import time.
