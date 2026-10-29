# Lip Reading Viseme Training & Setup Guide

This guide outlines the precise workflow used to set up the environment, prepare the multi-speaker datasets, and execute training inside Google Colab.

## Repository Setup & Installation
1. **Clone Repository**
   ```bash
   git clone https://github.com/Rathi4research/Lipreading_Viseme_New.git
   cd Lipreading_Viseme_New
   ```
2. **Install Python Packages**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download Dlib Landmarks Predictor**
   ```bash
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bunzip2 shape_predictor_68_face_landmarks.dat.bz2
   ```

## Dataset Structure & Extraction
The dataset must be structured under the `/content/Lipreading_Viseme_New/data` folder as follows:
```
data/
├── alignments/
│   ├── s1/
│   ├── s2/
│   └── s3/
└── videos/
    ├── s1/
    ├── s2/
    └── s3/
```

If extracting from zip archives stored on Google Drive (under `MyDrive/Training_Artifacts/`):
- Extract `s1_videos.zip`, `s2_videos.zip`, and `s3_videos.zip` directly into `data/videos/`.
- Extract `s1_alignments.zip`, `s2_alignments.zip`, and `s3_alignments.zip` directly into `data/alignments/`.
- Standardize the extracted directories so that folder names align perfectly (e.g. rename `s1_videos` to `s1` and `s1_alignments` to `s1`).

## Performance Optimization
To prevent the Google Colab session from crashing during dataset collation, **patch the data loader** to turn off multi-process worker overhead:
```python
# Inside src/data_loader.py, set num_workers=0
train_loader = DataLoader(..., num_workers=0)
```

## GPU Training Command
Before running the training command, ensure that **T4 GPU acceleration** is active under **Runtime > Change runtime type**.

Run the training across the 3 subjects:
```bash
python main.py train --model-type full --num-epochs 10 --batch-size 16 --train-subjects s1 --val-subjects s2 --test-subjects s3
```
