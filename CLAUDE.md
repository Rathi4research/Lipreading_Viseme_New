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

# Train (mode is positional; subjects/model options are flags)
python main.py train --model-type full --seq-model hybrid --visual-pool avg --gate intrinsic-extrinsic \
    --train-subjects s1 --val-subjects s2 --test-subjects s3

# Quick local run (e.g. Apple Silicon/MPS): checkpoints to ./checkpoints, few utterances per speaker
python main.py train --local-checkpoints --max-samples-per-subject 50 --num-epochs 2

# Resume an interrupted run (last_model.pth is written every epoch)
python main.py train ... --resume-from <checkpoint-dir>/last_model.pth

# Evaluate a checkpoint (model settings come from the checkpoint); --corrupt-ratio 0.3 = robustness test
python main.py evaluate --checkpoint-path <checkpoint-dir>/best_model.pth --test-subjects s3 --results-dir results

# Run the inference demo (prints reference vs greedy / grammar-constrained decoded sentences)
python main.py inference --checkpoint-path checkpoints/best_model.pth

# Print dataset statistics (frame/sequence shapes, viseme distribution, etc.)
python main.py analyze

# Generate a tiny synthetic dataset (GRID-format .align + random .npy + .meta.npz) for smoke-testing
python main.py create_sample --data-dir /tmp/smoke_data

# Regenerate the Colab notebook after editing its generator
python scripts/make_colab_notebook.py

# Standalone test of one utterance (.npy lip crops, or raw video if dlib is installed)
python test.py <checkpoint.pth> <file.npy|video> [speaker]
```

`main_old.py` is a stale duplicate of `main.py` — prefer `main.py`. `src/trainer.py` also has its own
`main()` for a minimal training run; `main.py train` is the primary entry point and supersedes it.

Runtime flags on `main.py` (all modes): `--device auto|cuda|mps|cpu` (auto = cuda > mps > cpu),
`--data-dir`, `--frame-size` (model input, default 88), `--no-augment`, `--num-workers` (default 2 on
CUDA, else 0), `--max-samples-per-subject`, `--batch-size`, `--learning-rate`, `--seed`, `--no-amp`,
`--min-detected-ratio`; model (train): `--model-type`, `--seq-model bilstm|transformer|hybrid|conformer`,
`--visual-pool avg|flatten`, `--gate ambiguity|entropy|learned|hard|intrinsic-extrinsic`; evaluate:
`--corrupt-ratio`, `--results-dir`. Checkpoint location for `train`: `Config.CHECKPOINT_DIR` (Google Drive on Colab) by
default, `--local-checkpoints` for `./checkpoints`, or `--checkpoint-dir PATH`. The trainer writes
`best_model.pth`, `last_model.pth` (every epoch), `checkpoint_epoch_N.pth` (every 5),
`training_metrics.csv` (one row per epoch) and `test_results.json` into that directory.

The Colab notebook (`notebooks/LipReading_Colab_Orchestrator.ipynb`, see `notebookrun.md`) is a thin
orchestrator (GPU/Drive setup, raw-data extraction, invoking `scripts/preprocess.py` and `main.py`, saving
outputs to Drive); code changes belong in the repo, not in notebook cells that patch source files.
`LipReading_Viseme_resolved_code.ipynb` is the older, superseded notebook that did patch files.

## Data layout

The `GRIDDataset` (`src/data_loader.py`) expects, per subject (e.g. `s1`, `s2`, `s3`):

```
data/alignments/<subject>/<utterance_id>.align        # "start end word" lines, GRID forced alignments
data/preprocessed_lips/<subject>/<utterance_id>.npy   # (T, 96, 96) uint8 mouth crops from scripts/preprocess.py
data/preprocessed_lips/<subject>/<utterance_id>.meta.npz  # landmarks, detection mask, crop transforms -> frame-quality signals
data/videos/<subject>/<utterance_id>.mpg|.mp4         # optional: only used if the .npy is missing
```

Samples are discovered from the `.npy` files and/or videos (an `.align` file is required).
`src/data_loader.py:prepare_frames` (shared by training, evaluation and `test.py`) turns stored crops
into model input: uint8 -> [0,1], crop to `Config.FRAME_HEIGHT x FRAME_WIDTH` (random position when
training, centre otherwise; smaller inputs such as old 80x80 float files are resized), normalise with
`Config.PIXEL_MEAN/PIXEL_STD`, and — training split only, fresh per sample — horizontal flip and time
masking (`Config.AUG_*`). Clips are then padded/truncated to `Config.SEQUENCE_LENGTH`.
Per clip the loader also returns the CTC character target (`src/decoding.py`), the real length, and
`frame_quality` signals from the `.meta.npz` ([missing/time-masked, gap to nearest detection, crop jitter],
0 = reliable); clips with a face detected in < `Config.MIN_DETECTED_RATIO` of frames are skipped, and
undetected frames are excluded from the per-frame losses.

Preprocessing (`scripts/preprocess.py`, default `--crop-mode aligned`): dlib 68 landmarks per frame,
missed frames linearly interpolated, landmarks smoothed over 12 frames (crop placement only), then a
per-frame similarity transform that levels the eye line, scales the crop to 1.4x the eye-centre
distance (face-sized, so mouth opening/rounding does not change the zoom) and centres the mouth corners. If an
utterance has no `.npy`, the raw video is cropped on the fly with dlib (slow; needs
`shape_predictor_68_face_landmarks.dat` at the repo root). None of this data is checked into the repo.

## Architecture

Data flows through these stages, wired together in `VisemePhonemeAdaptiveModel`
(`models/fusion_network.py`), the single top-level `nn.Module`. Model inputs are frames, clip lengths,
speaker ids and the preprocessing frame-quality signals — nothing derived from the targets (phonemes,
words, ground-truth ambiguity) may be fed in. `model.model_config` (`model_type`, `seq_model`,
`visual_pool`, `gate`) is stored in checkpoints; resume refuses a mismatch and `load_pretrained_model`
rebuilds the stored configuration, frame size and normalisation.

1. **Visual encoding** (`models/visual_encoder.py`) — `LipROI` (per-frame spatial attention gate) →
   `VisualEncoder` (4-block 3D CNN, spatial-only pooling so every frame keeps its own feature) →
   `--visual-pool avg` (spatial mean, default) or `flatten` (original, ~13M-param projection) → per-frame
   projection to `Config.VISUAL_FEATURE_DIM`.
2. **Sequence encoder** (`models/sequence_model.py`, `--seq-model`): `bilstm`, `transformer`, `hybrid`
   (BiLSTM → concat → Transformer, the original design) or `conformer`; all are padding-masked (LSTMs are
   packed by length). An auxiliary viseme head on its output (`aux_viseme_loss`) gives the detached viseme
   posterior that drives the intrinsic ambiguity score, the entropy signal and the LM input.
3. **Language model** (`src/language_model.py`) — `full`/`baseline` only. Input is the model's own soft
   phoneme estimate (aux viseme probs × viseme→phoneme prior). Phoneme Transformer → word BiLSTM
   (absolute frame positions) → cross-attention → speaker embedding (`model.speaker_ids_for`; unseen
   speakers use `Config.UNKNOWN_SPEAKER_ID`, trained via `Config.SPEAKER_DROPOUT`). Per-frame word logits.
4. **Fusion** (`src/adaptive_fusion.py`) — `full`: `MultiScaleAdaptiveFusion` (1/3/5/7-frame scales,
   scale attention) of 256-d visual and 512-d language features into 256-d fused features (no viseme
   bottleneck). Per-frame language weight from `--gate`: `ambiguity` (intrinsic ambiguity), `entropy`
   (posterior entropy), `learned` (cold-fusion style, from features), `hard` (threshold switch, CMTD
   style) or `intrinsic-extrinsic` (proposed: separate intrinsic-ambiguity and frame-quality channels whose
   logits add). `baseline`: concat → linear. `visual_only`: sequence features directly.
5. **Heads** on the fused features: `ctc_head` (28 classes: blank, space, a-z; main objective),
   `viseme_classifier` (auxiliary), `PriorGuidedPhonemeHead` (direct phoneme logits + learnable-scaled log of
   viseme probs × learnable P(phoneme | viseme) prior from `viseme_mapper`).

Loss (`compute_loss`, weights in `Config.LOSS_WEIGHTS`): CTC + viseme + aux viseme + phoneme + word
(not for visual_only) + gate-entropy regulariser (soft gates). Per-frame terms skip padding and frames
without a detected face and down-weight silence (`Config.SILENCE_LOSS_WEIGHT`). CTC runs on CPU under MPS.

`src/decoding.py`: greedy CTC decoding, exact best-path (Viterbi) decoding over the GRID grammar's
CTC graph (always a complete 6-word sentence), WER/CER. `src/metrics.py`: `run_evaluation` (shared by trainer and evaluator),
frame metrics (incl. phoneme accuracy within each viseme class), sentence metrics (per-slot accuracy,
letter-slot homophene groups derived from `viseme_mapper`, b/p d/t m/n confusions) and the gate analysis
(mean language weight by ambiguous × unreliable frame type).

`src/trainer.py`: AdamW + linear warmup + cosine decay (per step), mixed precision on CUDA, gradient
clipping; model selection and early stopping on validation greedy WER (tie-break val CTC loss); writes
`training_metrics.csv`, `best_model.pth`, `last_model.pth`, `test_results.json` (best checkpoint, full
metrics, commit and arguments). `src/evaluator.py` (`main.py evaluate`): report, JSON and confusion plots.

`src/viseme_mapping.py` (`viseme_mapper`) is the canonical phoneme↔viseme mapping and ambiguity source.

All tunable constants live in `src/config.py`'s `Config` class (class attributes). `main.py` mutates
`Config` from CLI flags before building datasets/models, so code reads `Config` at call/construction
time rather than capturing values at import time.
