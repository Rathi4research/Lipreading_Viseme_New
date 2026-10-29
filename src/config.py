import os
import torch


class Config:
    # Data paths
    DATA_DIR = "data"
    VIDEO_DIR = os.path.join(DATA_DIR, "videos")
    ALIGNMENT_DIR = os.path.join(DATA_DIR, "alignments")
    CHECKPOINT_DIR = "/content/drive/MyDrive/Lipreading_Checkpoints/"

    # Model parameters
    VISUAL_FEATURE_DIM = 512
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    NUM_HEADS = 8
    DROPOUT_RATE = 0.3

    # Video processing
    TARGET_FPS = 25
    FRAME_WIDTH = 96
    FRAME_HEIGHT = 96
    SEQUENCE_LENGTH = 75  # Max sequence length

    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    PATIENCE = 10

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

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
