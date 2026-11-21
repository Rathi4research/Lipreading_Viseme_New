import os
import torch


def detect_device():
    """Best available accelerator: CUDA (Colab/NVIDIA) > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Config:
    # Data paths
    DATA_DIR = "data"
    VIDEO_DIR = os.path.join(DATA_DIR, "videos")
    ALIGNMENT_DIR = os.path.join(DATA_DIR, "alignments")
    # Lip crops produced by scripts/preprocess.py: <PREPROCESSED_DIR>/<subject>/<utterance_id>.npy
    PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed_lips")

    # Checkpoints: CHECKPOINT_DIR is the persistent (Google Drive) default used on Colab;
    # LOCAL_CHECKPOINT_DIR is used instead when main.py is run with --local-checkpoints.
    CHECKPOINT_DIR = "/content/drive/MyDrive/Lipreading_Checkpoints/"
    LOCAL_CHECKPOINT_DIR = "checkpoints"

    @classmethod
    def set_data_dir(cls, data_dir):
        """Re-point all data paths at a new root (paths are derived, so update them together)."""
        cls.DATA_DIR = data_dir
        cls.VIDEO_DIR = os.path.join(data_dir, "videos")
        cls.ALIGNMENT_DIR = os.path.join(data_dir, "alignments")
        cls.PREPROCESSED_DIR = os.path.join(data_dir, "preprocessed_lips")

    # Model parameters
    VISUAL_FEATURE_DIM = 512
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    NUM_HEADS = 8
    DROPOUT_RATE = 0.3

    # Video processing
    TARGET_FPS = 25
    # Model input size. Crops are stored larger (PREPROCESS_SIZE, by scripts/preprocess.py) and the
    # data loader random-crops (training) or centre-crops (eval) this size out of them.
    FRAME_WIDTH = 88
    FRAME_HEIGHT = 88
    PREPROCESS_SIZE = 96
    SEQUENCE_LENGTH = 75  # Max sequence length

    # Pixel normalisation applied by the data loader after scaling to [0, 1]
    # (grayscale mouth-crop statistics used by the standard LRW/LRS3 recipes)
    PIXEL_MEAN = 0.421
    PIXEL_STD = 0.165

    # Training-time augmentation, drawn fresh for every sample, training split only
    AUGMENT = True
    AUG_FLIP_PROB = 0.5             # horizontal flip
    AUG_TIME_MASK_MAX_FRAMES = 10   # max length of each time mask
    AUG_TIME_MASKS_PER_SECOND = 1.0 # number of masks = this x clip duration (25 frames = 1 s)

    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-4            # peak learning rate (after warmup)
    WEIGHT_DECAY = 1e-2             # AdamW (decoupled) weight decay
    WARMUP_RATIO = 0.05             # share of all optimiser steps used for linear warmup
    MIN_LR_RATIO = 0.01             # cosine decay ends at this fraction of the peak
    GRAD_CLIP_NORM = 1.0
    NUM_EPOCHS = 100
    PATIENCE = 10                   # early stopping on the validation selection metric
    NUM_WORKERS = 0                 # DataLoader workers (main.py: 2 on CUDA by default)
    AMP = True                      # mixed precision; only takes effect on CUDA
    SEED = 42

    # Model structure (main.py --seq-model / --visual-pool / --gate)
    SEQ_MODEL = 'hybrid'            # bilstm | transformer | hybrid | conformer
    VISUAL_POOL = 'avg'             # avg (spatial average) | flatten (original, ~13M-param projection)
    GATE = 'intrinsic-extrinsic'    # ambiguity | entropy | learned | hard | intrinsic-extrinsic

    # Sentence-level CTC over characters (blank = index 0)
    CTC_CHARS = " abcdefghijklmnopqrstuvwxyz"
    CTC_BLANK = 0
    BEAM_SIZE = 16                  # grammar-constrained CTC prefix beam search

    # Loss weights. CTC is the main objective; the per-frame terms are auxiliary.
    LOSS_WEIGHTS = {
        'ctc': 1.0,
        'viseme': 0.5,
        'aux_viseme': 0.25,
        'phoneme': 0.5,
        'word': 0.25,               # forced to 0 for model_type visual_only
        'fusion_reg': 0.01,         # gate-entropy regulariser (full model, soft gates)
    }
    SILENCE_LOSS_WEIGHT = 0.3       # per-frame losses: weight of silence frames (~50% of GRID frames)

    # Frame quality (from scripts/preprocess.py .meta.npz)
    MIN_DETECTED_RATIO = 0.5        # skip clips where a face was found in fewer frames than this
    QUALITY_GAP_FRAMES = 10         # distance to the nearest detected frame is capped/normalised by this
    QUALITY_JITTER_PX = 4.0         # crop-centre jump (crop px) mapped to 1.0
    NUM_QUALITY_FEATURES = 3        # [missing/masked, gap to detection, crop jitter], 0 = reliable

    # Adaptive fusion parameters
    FUSION_ALPHA = 0.5  # Balance between visual and language model
    AMBIGUITY_THRESHOLD = 0.7  # Threshold for high ambiguity visemes

    # GRID corpus vocabulary
    COMMANDS = ['bin', 'lay', 'place', 'set']
    COLORS = ['blue', 'green', 'red', 'white']
    PREPOSITIONS = ['at', 'by', 'in', 'with']
    LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
               'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z']
    DIGITS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    ADVERBS = ['again', 'now', 'please', 'soon']

    # All vocabulary
    VOCABULARY = COMMANDS + COLORS + PREPOSITIONS + LETTERS + DIGITS + ADVERBS + ['sil']

    # Phoneme set (simplified for GRID corpus)
    PHONEMES = [
        'sil', 'b', 'd', 'g', 'p', 't', 'k', 'dx', 'q',
        'jh', 'ch', 's', 'sh', 'z', 'zh', 'f', 'th', 'v', 'dh',
        'm', 'n', 'ng', 'em', 'en', 'eng', 'nx', 'l', 'r', 'w', 'y',
        'hh', 'hv', 'el', 'iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw',
        'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw', 'ux', 'er', 'ax',
        'ix', 'axr', 'ax-h'
    ]

    VOCAB_SIZE = len(VOCABULARY)
    PHONEME_VOCAB_SIZE = len(PHONEMES)

    # GRID corpus speakers (s1-s34, s21 was never recorded/released)
    GRID_SPEAKERS = [f's{i}' for i in range(1, 35) if i != 21]
    SPEAKER_TO_ID = {speaker: idx for idx, speaker in enumerate(GRID_SPEAKERS)}
    NUM_SPEAKERS = len(GRID_SPEAKERS)
    # Extra speaker-embedding slot for speakers not seen in training (speaker-independent eval)
    UNKNOWN_SPEAKER_ID = NUM_SPEAKERS
    # Probability of replacing a training sample's speaker with UNKNOWN_SPEAKER_ID (trains that slot)
    SPEAKER_DROPOUT = 0.2

    # Device
    DEVICE = detect_device()
