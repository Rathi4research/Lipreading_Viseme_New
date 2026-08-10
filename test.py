import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import sys

# Add src to path for imports
sys.path.append('src')
sys.path.append('models')

from src.config import Config
from models.fusion_network import VisemePhonemeAdaptiveModel


def load_model(model_path):
    """Load the trained model"""
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None

    model = VisemePhonemeAdaptiveModel()

    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if 'epoch' in checkpoint:
                print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        print("✅ Model loaded successfully!")
        return model

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def process_video(video_path):
    """Process video and extract frames"""
    print(f"Processing video: {video_path}")

    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return None

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📹 Video: {total_frames} frames, {fps:.1f} FPS")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize to model input size
        frame = cv2.resize(frame, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))

        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        print("❌ No frames extracted from video")
        return None

    print(f"📹 Extracted {len(frames)} frames")

    frames = np.array(frames)

    # Handle sequence length
    if len(frames) > Config.SEQUENCE_LENGTH:
        start_idx = (len(frames) - Config.SEQUENCE_LENGTH) // 2
        frames = frames[start_idx:start_idx + Config.SEQUENCE_LENGTH]
        print(f"⚠️  Trimmed to {Config.SEQUENCE_LENGTH} frames")
    elif len(frames) < Config.SEQUENCE_LENGTH:
        last_frame = frames[-1]
        padding_needed = Config.SEQUENCE_LENGTH - len(frames)
        padding = np.tile(last_frame[np.newaxis, :, :], (padding_needed, 1, 1))
        frames = np.concatenate([frames, padding], axis=0)
        print(f"📝 Padded to {Config.SEQUENCE_LENGTH} frames")

    # Add batch dimension: (1, seq_len, height, width)
    video_tensor = torch.FloatTensor(frames).unsqueeze(0)

    return video_tensor


def predict_sentence(model, video_tensor):
    model.eval()
    with torch.no_grad():
        # Correct shape unpacking
        batch, seq_len, height, width = video_tensor.shape

        pad_idx = 0
        phoneme_targets = torch.full((batch, seq_len), pad_idx, dtype=torch.long)
        word_targets    = torch.full((batch, seq_len), pad_idx, dtype=torch.long)

        outputs = model(
            video_tensor,
            phoneme_targets,   # phoneme_targets
            word_targets,      # word_targets
            None,              # speaker_ids
            None               # ambiguity_scores
        )

        viseme_logits  = outputs['viseme_logits']
        phoneme_logits = outputs['phoneme_logits']

        # Simple prediction
        phoneme_preds = torch.argmax(phoneme_logits, dim=-1)

        # Dummy confidence placeholders for now
        return phoneme_preds, 0.0, []



def test_video_prediction(model_path, video_path):
    """Main function to test video prediction"""
    print("🎬 Manual Video Prediction Test")
    print("=" * 50)

    model = load_model(model_path)
    if model is None:
        return

    video_tensor = process_video(video_path)
    if video_tensor is None:
        return

    predicted_sentence, avg_confidence, confidence_scores = predict_sentence(model, video_tensor)

    print("\n🎯 PREDICTION RESULTS:")
    print("=" * 50)
    print(f"📹 Video: {os.path.basename(video_path)}")
    print(f"🗣️  Predicted Sentence: '{predicted_sentence}'")
    print(f"📊 Average Confidence: {avg_confidence:.3f}")
    print(f"📊 Word Confidences: {[f'{c:.3f}' for c in confidence_scores]}")

    if avg_confidence > 0.8:
        print("✅ High confidence prediction!")
    elif avg_confidence > 0.6:
        print("⚠️  Medium confidence prediction")
    else:
        print("❌ Low confidence prediction")

    print("=" * 50)


if __name__ == "__main__":
    MODEL_PATH = r"D:\lip_reading_project\checkpoints\best_model.pth"   # Change if needed
    VIDEO_PATH = r"D:\lip_reading_project\data\videos\s1\bbaf2n.mpg"    # Change if needed

    if len(sys.argv) >= 3:
        MODEL_PATH = sys.argv[1]
        VIDEO_PATH = sys.argv[2]
    elif len(sys.argv) == 2:
        print("Usage: python test.py <model_path> <video_path>")
        sys.exit(1)

    print(f"Model: {MODEL_PATH}")
    print(f"Video: {VIDEO_PATH}")
    print()

    test_video_prediction(MODEL_PATH, VIDEO_PATH)

    # Option to test multiple videos interactively
    while True:
        print("\n" + "=" * 50)
        choice = input("🎬 Test another video? (y/n): ").lower().strip()

        if choice in ['y', 'yes']:
            new_video = input("📹 Enter video path: ").strip()
            if os.path.exists(new_video):
                test_video_prediction(MODEL_PATH, new_video)
            else:
                print(f"❌ Video not found: {new_video}")
        elif choice in ['n', 'no']:
            print("👋 Goodbye!")
            break
        else:
            print("Please enter 'y' or 'n'")
