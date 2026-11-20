"""Manual single-utterance test of a trained checkpoint.

Usage:
    python test.py <checkpoint.pth> <input>

<input> is either a preprocessed lip-crop .npy (from scripts/preprocess.py, preferred) or a raw
GRID video (.mpg/.mp4), which is lip-cropped the same way as in preprocessing (needs dlib and
shape_predictor_68_face_landmarks.dat). Feeding whole, uncropped frames would not match what the
model was trained on.
"""
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from src.config import Config, detect_device
from src.data_loader import prepare_frames
from models.fusion_network import ModelFactory


def load_model(model_path):
    """Load a trained model (model_type, frame size and known speakers come from the checkpoint)"""
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = ModelFactory.load_pretrained_model(model_path)  # also applies the checkpoint's frame size
    model.eval()
    print(f"Loaded '{model.model_type}' model (epoch {checkpoint.get('epoch', -1) + 1}, "
          f"frames {Config.FRAME_HEIGHT}x{Config.FRAME_WIDTH})")
    return model


def load_input(path, landmarks='shape_predictor_68_face_landmarks.dat'):
    """Return model-ready frames (1, T, 1, H, W), padded to SEQUENCE_LENGTH, plus the real length.

    Uses the data loader's prepare_frames (centre crop + normalisation, no augmentation), so the
    input matches what the model saw during evaluation.
    """
    if not os.path.exists(path):
        print(f"Input not found: {path}")
        return None, 0

    if path.endswith('.npy'):
        frames = np.load(path)
    else:
        from scripts import preprocess
        if preprocess.dlib is None:
            print("dlib is required to crop raw video; use a preprocessed .npy instead")
            return None, 0
        preprocess._init_worker(landmarks)
        frames = np.asarray(preprocess.extract_lip_sequence(path, size=Config.PREPROCESS_SIZE,
                                                            max_frames=Config.SEQUENCE_LENGTH), dtype=np.uint8)

    frames = frames[:Config.SEQUENCE_LENGTH]
    num_valid = len(frames)
    video = prepare_frames(frames, augment=False)  # (T, 1, H, W)
    if num_valid < Config.SEQUENCE_LENGTH:
        pad = torch.zeros(Config.SEQUENCE_LENGTH - num_valid, *video.shape[1:])
        video = torch.cat([video, pad], dim=0)
    return video.unsqueeze(0), num_valid


def collapse(labels):
    """Merge consecutive repeats and drop silence"""
    out = []
    for label in labels:
        if not out or out[-1] != label:
            out.append(label)
    return [label for label in out if label not in ('sil', 'V11')]


def predict(model, video, num_valid, speaker=None):
    device = next(model.parameters()).device
    speaker_ids = model.speaker_ids_for([speaker or 'unknown'], device)
    preds = model.predict(video.to(device), speaker_ids=speaker_ids)

    phoneme_ids = preds['phoneme_predictions'][0, :num_valid].tolist()
    viseme_ids = preds['viseme_predictions'][0, :num_valid].tolist()
    confidence = preds['phoneme_probabilities'][0, :num_valid].max(dim=-1).values.mean().item()

    phonemes = collapse([Config.PHONEMES[i] for i in phoneme_ids])
    visemes = collapse([f"V{i}" for i in viseme_ids])
    return phonemes, visemes, confidence


def test_prediction(model, input_path, speaker=None):
    video, num_valid = load_input(input_path)
    if video is None:
        return
    phonemes, visemes, confidence = predict(model, video, num_valid, speaker)

    print("\nPREDICTION RESULTS")
    print("=" * 50)
    print(f"Input: {os.path.basename(input_path)} ({num_valid} frames)")
    print(f"Phonemes: {' '.join(phonemes)}")
    print(f"Visemes:  {' '.join(visemes)}")
    print(f"Mean per-frame phoneme confidence: {confidence:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    model_path, input_path = sys.argv[1], sys.argv[2]
    speaker = sys.argv[3] if len(sys.argv) > 3 else None  # optional, e.g. s1

    Config.DEVICE = detect_device()
    model = load_model(model_path)
    if model is None:
        sys.exit(1)
    model.to(Config.DEVICE)

    test_prediction(model, input_path, speaker)

    while sys.stdin.isatty():
        choice = input("\nTest another input? (y/n): ").lower().strip()
        if choice in ('y', 'yes'):
            test_prediction(model, input("Path to .npy or video: ").strip(), speaker)
        elif choice in ('n', 'no'):
            break
