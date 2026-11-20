# Running on Google Colab

Use **`notebooks/LipReading_Colab_Orchestrator.ipynb`**. It only orchestrates: it clones the repo, then runs
`scripts/preprocess.py` and `main.py` with command-line flags. It never edits repository files, so every
change to the pipeline belongs in the code, not in notebook cells.

(`LipReading_Viseme_resolved_code.ipynb` is the older notebook that patched source files at runtime; it is
kept for reference only and does not match the current code.)

## Steps
1. Open the notebook in Colab and select a GPU runtime (*Runtime → Change runtime type → T4 GPU*).
2. Edit **1. Settings**:
   - `BRANCH` — the branch to run (default `advanced-preprocessing`).
   - `RAW_DATA_DIR` — Drive folder with the raw GRID archives (`sN_videos.zip` / `sN.mpg_vcd.zip` for videos,
     `sN.tar` / `sN_alignments.zip` for alignments). Default `MyDrive/Training_Artifacts`.
   - `OUTPUT_DIR` — Drive folder for everything produced. Default `MyDrive/Lipreading_advanced_preprocessing`.
   - `PREPROCESS_SPEAKERS`, crop settings, and the train/val/test speakers and hyperparameters.
3. *Runtime → Run all*. After a disconnect, Run all again: finished speakers are skipped and training resumes
   from the run's `last_model.pth`.

## What ends up on Drive
```
<OUTPUT_DIR>/
├── preprocessed/<crop-mode>_scale<s>_<size>px/sN_preprocessed.tar.gz   # .npy crops + .meta.npz landmarks
├── alignments/sN_alignments.tar.gz
└── runs/<RUN_NAME>/   best_model.pth, last_model.pth, training_metrics.csv, test_results.json,
                       training_curves.png, evaluation/, lipreading.log, tensorboard/, run_config.json
```

## Using the preprocessed data locally
```bash
mkdir -p data/preprocessed_lips data/alignments
tar -xzf sN_preprocessed.tar.gz -C data/preprocessed_lips
tar -xzf sN_alignments.tar.gz   -C data/alignments
python main.py train --local-checkpoints --train-subjects s1 --val-subjects s2 --test-subjects s2 --num-epochs 2
```
