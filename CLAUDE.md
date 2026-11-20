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
# dlib is only needed for scripts/preprocess.py (and the slow on-the-fly fallback in the data loader)

# Pre-extract mouth crops once (parallel dlib) -> data/preprocessed_lips/<speaker>/<utt>.npy (+ .meta.npz)
# Default: face-aligned 96x96 uint8 crops; --crop-mode square|tight for the older crops (ablations)
python scripts/preprocess.py --speakers s1 s2 s3 [--backup-dir <drive folder>]

# Train (mode is positional; subjects/model-type/etc. are flags)
python main.py train --model-type full --train-subjects s1 --val-subjects s2 --test-subjects s3

# Quick local run (e.g. Apple Silicon/MPS): checkpoints to ./checkpoints, few utterances per speaker
python main.py train --local-checkpoints --max-samples-per-subject 50 --num-epochs 2

# Resume an interrupted run (last_model.pth is written every epoch)
python main.py train ... --resume-from <checkpoint-dir>/last_model.pth

# Evaluate a checkpoint
python main.py evaluate --checkpoint-path <checkpoint-dir>/best_model.pth --model-type full

# Run the inference demo (prints per-sample viseme/phoneme predictions)
python main.py inference --checkpoint-path checkpoints/best_model.pth

# Print dataset statistics (frame/sequence shapes, viseme distribution, etc.)
python main.py analyze

# Generate a tiny synthetic dataset (GRID-format .align + random .npy) for smoke-testing
python main.py create_sample --data-dir /tmp/smoke_data --frame-size 80

# Standalone test of one utterance (.npy lip crops, or raw video if dlib is installed)
python test.py <checkpoint.pth> <file.npy|video> [speaker]
```

`main_old.py` is a stale duplicate of `main.py` — prefer `main.py`. `src/trainer.py` also has its own
`main()` for a minimal training run; `main.py train` is the primary entry point and supersedes it.

Runtime flags on `main.py` (all modes): `--device auto|cuda|mps|cpu` (auto = cuda > mps > cpu),
`--data-dir`, `--frame-size` (model input, default 88), `--no-augment`, `--num-workers`,
`--max-samples-per-subject`, `--batch-size`, `--learning-rate`. Checkpoint location for `train`: `Config.CHECKPOINT_DIR` (Google Drive on Colab) by
default, `--local-checkpoints` for `./checkpoints`, or `--checkpoint-dir PATH`. The trainer writes
`best_model.pth`, `last_model.pth` (every epoch), `checkpoint_epoch_N.pth` (every 5),
`training_metrics.csv` (one row per epoch) and `test_results.json` into that directory.

The Colab notebook (`LipReading_Viseme_resolved_code.ipynb`) is meant to be a thin orchestrator
(GPU/Drive setup, data restore, invoking `scripts/preprocess.py` and `main.py`); code changes belong
in the repo, not in notebook cells that patch source files.

## Data layout

The `GRIDDataset` (`src/data_loader.py`) expects, per subject (e.g. `s1`, `s2`, `s3`):

```
data/alignments/<subject>/<utterance_id>.align        # "start end word" lines, GRID forced alignments
data/preprocessed_lips/<subject>/<utterance_id>.npy   # (T, 96, 96) uint8 mouth crops from scripts/preprocess.py
data/preprocessed_lips/<subject>/<utterance_id>.meta.npz  # landmarks, detection mask, crop transforms (not yet used)
data/videos/<subject>/<utterance_id>.mpg|.mp4         # optional: only used if the .npy is missing
```

Samples are discovered from the `.npy` files and/or videos (an `.align` file is required).
`src/data_loader.py:prepare_frames` (shared by training, evaluation and `test.py`) turns stored crops
into model input: uint8 -> [0,1], crop to `Config.FRAME_HEIGHT x FRAME_WIDTH` (random position when
training, centre otherwise; smaller inputs such as old 80x80 float files are resized), normalise with
`Config.PIXEL_MEAN/PIXEL_STD`, and — training split only, fresh per sample — horizontal flip and time
masking (`Config.AUG_*`). Clips are then padded/truncated to `Config.SEQUENCE_LENGTH`.

Preprocessing (`scripts/preprocess.py`, default `--crop-mode aligned`): dlib 68 landmarks per frame,
missed frames linearly interpolated, landmarks smoothed over 12 frames (crop placement only), then a
per-frame similarity transform that levels the eye line, scales the crop to 1.4x the eye-centre
distance (face-sized, so mouth opening/rounding does not change the zoom) and centres the mouth corners. If an
utterance has no `.npy`, the raw video is cropped on the fly with dlib (slow; needs
`shape_predictor_68_face_landmarks.dat` at the repo root). None of this data is checked into the repo.

## Architecture

Data flows through these stages, wired together in `VisemePhonemeAdaptiveModel`
(`models/fusion_network.py`), the single top-level `nn.Module`. Only frames and speaker ids are model
inputs — nothing derived from the targets (phonemes, words, ground-truth ambiguity) may be fed in.

1. **Visual encoding** (`models/visual_encoder.py`) — `EnhancedVisualEncoder` = `LipROI` (a 2D
   spatial-attention gate applied per-frame) → `VisualEncoder` (4-block 3D CNN; pooling is spatial only,
   so all `SEQUENCE_LENGTH` frames keep their own feature vector) → per-frame `fc1`/`fc2` projection to
   `Config.VISUAL_FEATURE_DIM`. All layers are built in `__init__` (sized from `Config.FRAME_*` at
   construction time). `upgrade_legacy_state_dict` maps old checkpoints (`fc1_dynamic`) onto this layout.

2. **Sequence modeling** (`models/sequence_model.py`) — `HybridSequenceModel` (LSTM → concat with input →
   Transformer with sinusoidal positions) gives per-frame hidden features (`Config.HIDDEN_DIM`) and its own
   *auxiliary* viseme logits, trained with `aux_viseme_loss` (`Config.AUX_VISEME_LOSS_WEIGHT`). The
   detached aux viseme distribution drives the ambiguity scores and the language-model input below.

3. **Language modeling** (`src/language_model.py`) — runs for `model_type` `full` and `baseline`. Input is
   the model's own soft phoneme estimate (aux viseme probs × viseme→phoneme prior), never the ground truth.
   `ContextualLanguageModel` wraps `HierarchicalLanguageModel` (`PhonemeLanguageModel` transformer with
   positional encoding → `GRIDLanguageModel` word LSTM with absolute frame-position embeddings,
   cross-attended) and adds a speaker embedding. Speakers come from `model.speaker_ids_for(subjects)`;
   speakers not in `model.known_speakers` (the training speakers, stored in checkpoints) use the extra
   `Config.UNKNOWN_SPEAKER_ID` slot, which is trained via `Config.SPEAKER_DROPOUT`. Its per-frame word
   logits are trained with `word_loss` against per-frame word labels.

4. **Fusion** (`src/adaptive_fusion.py`) — `full`: `MultiScaleAdaptiveFusion` runs `AdaptiveFusionModule`
   at 4 temporal scales (1/3/5/7 frames), gating visual vs. language features per frame from the ambiguity
   score (expected `viseme_mapper` ambiguity under the aux viseme distribution), combined by learned
   scale-attention; output (num_visemes) → `fused_proj` → HIDDEN_DIM. `baseline`: plain `static_fusion`
   (concat → linear). `visual_only`: no language model, sequence features used directly.

5. **Viseme→phoneme decoding** (`src/adaptive_fusion.py`) — `ConfusionMatrixDecoder`: prior
   P(phoneme | viseme) from `viseme_mapper.confusion_matrix` (12 × len(`Config.PHONEMES`), columns in
   `Config.PHONEMES` order = target indices), adjusted in log space by a learnable matrix; viseme probs ×
   prior gives phoneme probs, and a BiLSTM adds a zero-initialised residual correction to their logs.

`src/viseme_mapping.py` (`viseme_mapper`, a module-level singleton) is the shared source of truth for the
phoneme↔viseme mapping and ambiguity scores; treat it as canonical rather than re-deriving mappings.

`model_type` (`full` / `baseline` / `visual_only`, built via `ModelFactory.create`) is stored on the
model and in checkpoints; training, resume (`load_checkpoint` refuses a mismatch) and evaluation
(`load_pretrained_model` uses the checkpoint's type and frame size) all run the same path.

Labels: the data loader returns per-frame `phonemes`, `visemes`, `word_labels` (padding frames past the
end of the video = -1, ignored everywhere) and ground-truth `ambiguity_scores` (metrics only).

`src/trainer.py`'s `LipReadingTrainer` handles the training loop, TensorBoard logging (`runs/`),
checkpointing (see the checkpoint directory flags above; gradient clipping at norm 1.0,
`ReduceLROnPlateau`, early stopping via `Config.PATIENCE`) and a final test evaluation of the **best**
checkpoint. Metrics are reported on all valid frames and on speech-only frames (silence excluded, since
it dominates GRID clips), plus a high/low ground-truth-ambiguity split. `src/evaluator.py`'s
`LipReadingEvaluator` is the more comprehensive standalone path used by `main.py evaluate`, producing
confusion matrix plots (`plots/`) and a text report (`results/`).

All tunable constants (paths, model dims, video/frame params, training hyperparameters, the GRID
vocabulary/phoneme set) live in one place: `src/config.py`'s `Config` class (accessed as class attributes,
not instantiated). `main.py` mutates `Config` at runtime from CLI flags (`DEVICE`, `BATCH_SIZE`,
`FRAME_*`, data paths, ...) before building datasets/models, so code reads `Config` at call/construction
time rather than capturing values at import time.
