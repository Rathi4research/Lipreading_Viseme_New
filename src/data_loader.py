import os
import logging
import torch
import cv2
import numpy as np
import dlib
from torch.utils.data import Dataset, DataLoader
from src.config import Config
from src.viseme_mapping import viseme_mapper

logger = logging.getLogger(__name__)

# GRID's .align files label each interval with a whole word (or 'sil'/'sp'), not a
# phoneme. Config.PHONEMES/viseme_mapper only understand phonemes, so each word in the
# closed GRID vocabulary is expanded to its phone sequence (TIMIT-style, matching
# Config.PHONEMES) before phoneme/viseme label lookup. Without this, every word silently
# missed the phoneme lookup and defaulted to 'sil'/silence-viseme, collapsing nearly all
# labels to a single class.
GRID_WORD_TO_PHONEMES = {
    'sil': ['sil'],
    # Commands
    'bin': ['b', 'ih', 'n'],
    'lay': ['l', 'ey'],
    'place': ['p', 'l', 'ey', 's'],
    'set': ['s', 'eh', 't'],
    # Colors
    'blue': ['b', 'l', 'uw'],
    'green': ['g', 'r', 'iy', 'n'],
    'red': ['r', 'eh', 'd'],
    'white': ['w', 'ay', 't'],
    # Prepositions
    'at': ['ae', 't'],
    'by': ['b', 'ay'],
    'in': ['ih', 'n'],
    'with': ['w', 'ih', 'th'],
    # Letters (a-z, excluding 'w')
    'a': ['ey'], 'b': ['b', 'iy'], 'c': ['s', 'iy'], 'd': ['d', 'iy'],
    'e': ['iy'], 'f': ['eh', 'f'], 'g': ['jh', 'iy'], 'h': ['ey', 'ch'],
    'i': ['ay'], 'j': ['jh', 'ey'], 'k': ['k', 'ey'], 'l': ['eh', 'l'],
    'm': ['eh', 'm'], 'n': ['eh', 'n'], 'o': ['ow'], 'p': ['p', 'iy'],
    'q': ['k', 'y', 'uw'], 'r': ['aa', 'r'], 's': ['eh', 's'], 't': ['t', 'iy'],
    'u': ['y', 'uw'], 'v': ['v', 'iy'], 'x': ['eh', 'k', 's'], 'y': ['w', 'ay'],
    'z': ['z', 'eh', 'd'],
    # Digits
    'zero': ['z', 'iy', 'r', 'ow'], 'one': ['w', 'ah', 'n'], 'two': ['t', 'uw'],
    'three': ['th', 'r', 'iy'], 'four': ['f', 'ao', 'r'], 'five': ['f', 'ay', 'v'],
    'six': ['s', 'ih', 'k', 's'], 'seven': ['s', 'eh', 'v', 'ax', 'n'],
    'eight': ['ey', 't'], 'nine': ['n', 'ay', 'n'],
    # Adverbs
    'again': ['ax', 'g', 'eh', 'n'], 'now': ['n', 'aw'],
    'please': ['p', 'l', 'iy', 'z'], 'soon': ['s', 'uw', 'n'],
}


class GRIDDataset(Dataset):
    def __init__(self, data_dir, subjects=['s1', 's2', 's3'], transform=None):
        self.data_dir = data_dir
        self.subjects = subjects
        self.transform = transform
        self.samples = []

        # dlib face detector + landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # Load all video-alignment pairs
        for subject in subjects:
            video_dir = os.path.join(Config.VIDEO_DIR, subject)
            align_dir = os.path.join(Config.ALIGNMENT_DIR, subject)

            if os.path.exists(video_dir) and os.path.exists(align_dir):
                video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.mpg'))]

                for video_file in video_files:
                    base_name = os.path.splitext(video_file)[0]
                    align_file = base_name + '.align'

                    video_path = os.path.join(video_dir, video_file)
                    align_path = os.path.join(align_dir, align_file)

                    if os.path.exists(align_path):
                        self.samples.append({
                            'video_path': video_path,
                            'align_path': align_path,
                            'subject': subject,
                            'utterance_id': base_name
                        })

        print(f"Loaded {len(self.samples)} samples from subjects {subjects}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load video frames (mouth cropped)
        frames = self.load_video(sample['video_path'])

        # Load alignment (phoneme sequence with timestamps)
        phonemes, words, visemes = self.load_alignment(sample['align_path'])

        # Convert to tensors
        frames_tensor = torch.FloatTensor(frames)
        phonemes_tensor = torch.LongTensor(phonemes)
        visemes_tensor = torch.LongTensor(visemes)

        # Calculate ambiguity scores
        ambiguity_scores = [viseme_mapper.get_ambiguity_score(v) for v in visemes]
        ambiguity_tensor = torch.FloatTensor(ambiguity_scores)

        return {
            'frames': frames_tensor,
            'phonemes': phonemes_tensor,
            'visemes': visemes_tensor,
            'ambiguity_scores': ambiguity_tensor,
            'words': words,
            'subject': sample['subject'],
            'utterance_id': sample['utterance_id']
        }

    def load_video(self, video_path):
        """Load video frames and crop mouth region"""
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = self.detector(gray, 1)

            if len(dets) > 0:
                shape = self.predictor(gray, dets[0])
                mouth_points = shape.parts()[48:68]  # landmarks 48–67 are mouth
                x_min = min([p.x for p in mouth_points])
                y_min = min([p.y for p in mouth_points])
                x_max = max([p.x for p in mouth_points])
                y_max = max([p.y for p in mouth_points])

                # crop mouth ROI
                mouth_roi = gray[y_min:y_max, x_min:x_max]
                mouth_roi = cv2.resize(mouth_roi, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))
                mouth_roi = mouth_roi.astype(np.float32) / 255.0
                frames.append(mouth_roi)
            else:
                # fallback: use whole resized frame
                frame = cv2.resize(gray, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)

        cap.release()

        # Pad or truncate to fixed length
        frames = np.array(frames)

        if len(frames) > Config.SEQUENCE_LENGTH:
            frames = frames[:Config.SEQUENCE_LENGTH]
        elif len(frames) < Config.SEQUENCE_LENGTH:
            padding = np.zeros((Config.SEQUENCE_LENGTH - len(frames),
                                Config.FRAME_HEIGHT, Config.FRAME_WIDTH))
            frames = np.concatenate([frames, padding], axis=0)

        return frames

    def load_alignment(self, align_path):
        """Build a per-frame phoneme/viseme sequence aligned to the video's frame grid.

        GRID .align timestamps are given in units of 1/25000 sec; dividing by 1000
        converts them to a frame index at the corpus's native 25 fps (Config.TARGET_FPS,
        matching Config.SEQUENCE_LENGTH=75 for ~3s clips). Each word interval is mapped to
        its covered frame range, and the word's phones (see GRID_WORD_TO_PHONEMES) are
        spread evenly across that range, so frame t's label is the phone actually being
        spoken at that instant rather than a position in a word-count-length list that had
        no relation to the video's actual timing.
        """
        intervals = []  # (start_frame, end_frame, word_label)

        with open(align_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 3:
                        label = parts[2]
                        if label != 'sp':  # skip pauses
                            start_frame = int(parts[0]) // 1000
                            end_frame = int(parts[1]) // 1000
                            intervals.append((start_frame, end_frame, label))

        words = [label for _, _, label in intervals if label in Config.VOCABULARY]

        num_frames = Config.SEQUENCE_LENGTH
        frame_phonemes = ['sil'] * num_frames

        for start_frame, end_frame, label in intervals:
            word_phones = GRID_WORD_TO_PHONEMES.get(label)
            if word_phones is None:
                logger.warning(
                    f"No pronunciation for word '{label}' in {align_path}; "
                    f"treating as silence"
                )
                word_phones = ['sil']

            span = max(end_frame - start_frame, 1)
            for t in range(max(start_frame, 0), min(end_frame, num_frames)):
                phone_idx = min(int((t - start_frame) / span * len(word_phones)),
                                 len(word_phones) - 1)
                frame_phonemes[t] = word_phones[phone_idx]

        # Convert per-frame phonemes to indices
        phoneme_to_idx = {p: i for i, p in enumerate(Config.PHONEMES)}
        phoneme_indices = [phoneme_to_idx.get(p, 0) for p in frame_phonemes]

        # Convert to visemes
        viseme_indices = viseme_mapper.phoneme_to_viseme_sequence(frame_phonemes)

        return phoneme_indices, words, viseme_indices


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    frames = torch.stack([item['frames'] for item in batch])
    phonemes = torch.stack([item['phonemes'] for item in batch])
    visemes = torch.stack([item['visemes'] for item in batch])
    ambiguity_scores = torch.stack([item['ambiguity_scores'] for item in batch])

    return {
        'frames': frames,
        'phonemes': phonemes,
        'visemes': visemes,
        'ambiguity_scores': ambiguity_scores,
        'words': [item['words'] for item in batch],
        'subjects': [item['subject'] for item in batch],
        'utterance_ids': [item['utterance_id'] for item in batch]
    }


def create_data_loaders(train_subjects=['s1'], val_subjects=['s2'], test_subjects=['s3']):
    """Create train, validation, and test data loaders"""

    train_dataset = GRIDDataset(Config.DATA_DIR, subjects=train_subjects)
    val_dataset = GRIDDataset(Config.DATA_DIR, subjects=val_subjects)
    test_dataset = GRIDDataset(Config.DATA_DIR, subjects=test_subjects)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE,
                             shuffle=False, collate_fn=collate_fn, num_workers=4)

    return train_loader, val_loader, test_loader
