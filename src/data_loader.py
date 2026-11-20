import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
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
    """GRID utterances as (frames, per-frame phoneme/viseme labels).

    Frames come from the preprocessed lip crops in Config.PREPROCESSED_DIR
    (<subject>/<utterance_id>.npy, produced by scripts/preprocess.py). If an utterance has no
    .npy file, the raw video in Config.VIDEO_DIR is cropped on the fly with dlib instead (slow).
    Samples are discovered from either source, so a local run only needs the .npy files and
    the .align files — the raw videos are optional.
    """

    def __init__(self, data_dir, subjects=['s1', 's2', 's3'], transform=None, max_samples_per_subject=None):
        self.data_dir = data_dir
        self.subjects = subjects
        self.transform = transform
        self.samples = []

        # dlib detector/predictor are only needed for the on-the-fly fallback; created lazily.
        self._detector = None
        self._predictor = None
        self._warned_fallback = False

        for subject in subjects:
            video_dir = os.path.join(Config.VIDEO_DIR, subject)
            npy_dir = os.path.join(Config.PREPROCESSED_DIR, subject)
            align_dir = os.path.join(Config.ALIGNMENT_DIR, subject)

            if not os.path.isdir(align_dir):
                logger.warning(f"No alignment directory for subject {subject}: {align_dir}")
                continue

            # utterance_id -> raw video path (if any)
            videos = {}
            if os.path.isdir(video_dir):
                for f in os.listdir(video_dir):
                    if f.endswith(('.mp4', '.mpg')):
                        videos[os.path.splitext(f)[0]] = os.path.join(video_dir, f)

            npys = set()
            if os.path.isdir(npy_dir):
                npys = {os.path.splitext(f)[0] for f in os.listdir(npy_dir) if f.endswith('.npy')}

            subject_samples = []
            for utterance_id in sorted(set(videos) | npys):
                align_path = os.path.join(align_dir, utterance_id + '.align')
                if not os.path.exists(align_path):
                    continue
                subject_samples.append({
                    'video_path': videos.get(utterance_id),
                    'npy_path': os.path.join(npy_dir, utterance_id + '.npy') if utterance_id in npys else None,
                    'align_path': align_path,
                    'subject': subject,
                    'utterance_id': utterance_id
                })

            if max_samples_per_subject is not None:
                subject_samples = subject_samples[:max_samples_per_subject]
            self.samples.extend(subject_samples)

        print(f"Loaded {len(self.samples)} samples from subjects {subjects}")

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, sample):
        """Return frames as a float tensor of shape (T, 1, H, W)."""
        if sample['npy_path'] is not None:
            frames = np.load(sample['npy_path'])
        elif sample['video_path'] is not None:
            if not self._warned_fallback:
                logger.warning("No preprocessed .npy for some samples (e.g. %s); cropping raw video "
                               "with dlib on the fly, which is slow. Run scripts/preprocess.py first.",
                               sample['utterance_id'])
                self._warned_fallback = True
            frames = self.load_video(sample['video_path'])
        else:
            raise FileNotFoundError(f"No .npy or video found for {sample['subject']}/{sample['utterance_id']}")

        return torch.tensor(frames, dtype=torch.float32).unsqueeze(1)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        align_path = sample['align_path']

        video_tensor = self._load_frames(sample)

        # Pad or truncate sequence length to Config.SEQUENCE_LENGTH
        seq_len = video_tensor.shape[0]
        expected_seq_len = Config.SEQUENCE_LENGTH
        if seq_len < expected_seq_len:
            # Pad with zeros at the end
            padding = torch.zeros((expected_seq_len - seq_len, 1, video_tensor.shape[2], video_tensor.shape[3]), dtype=torch.float32)
            video_tensor = torch.cat((video_tensor, padding), dim=0)
        elif seq_len > expected_seq_len:
            # Truncate
            video_tensor = video_tensor[:expected_seq_len]

        # Force resize spatial dimensions to match the config
        expected_h, expected_w = Config.FRAME_HEIGHT, Config.FRAME_WIDTH
        if video_tensor.shape[-2:] != (expected_h, expected_w):
            video_tensor = F.interpolate(video_tensor, size=(expected_h, expected_w), mode='bilinear', align_corners=False)

        # Load alignment (phoneme sequence, words, visemes)
        phonemes, words, visemes = self.load_alignment(align_path)

        # Convert to tensors
        phonemes_tensor = torch.LongTensor(phonemes)
        visemes_tensor = torch.LongTensor(visemes)

        # Calculate ambiguity scores
        ambiguity_scores = [viseme_mapper.get_ambiguity_score(v) for v in visemes]
        ambiguity_tensor = torch.FloatTensor(ambiguity_scores)

        # Apply transforms if provided
        if self.transform:
            video_tensor = self.transform(video_tensor)

        return {
            'frames': video_tensor,
            'phonemes': phonemes_tensor,
            'visemes': visemes_tensor,
            'ambiguity_scores': ambiguity_tensor,
            'words': words,
            'subject': sample['subject'],
            'utterance_id': sample['utterance_id']
        }

    def load_video(self, video_path):
        """Load video frames and crop mouth region (on-the-fly fallback; needs dlib + cv2)"""
        import cv2
        import dlib

        if self._detector is None:
            self._detector = dlib.get_frontal_face_detector()
            self._predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = self._detector(gray, 1)

            if len(dets) > 0:
                shape = self._predictor(gray, dets[0])
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
                            if len(parts) >= 2 and not (parts[0].isdigit() and parts[1].isdigit()):
                                continue
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
    # Ensure all frames have the same sequence length by padding to max_len in batch
    max_len = max([item['frames'].shape[0] for item in batch])

    processed_frames = []
    for item in batch:
        f = item['frames']
        if f.shape[0] < max_len:
            # Pad along the temporal dimension (dim 0)
            pad = torch.zeros((max_len - f.shape[0], f.shape[1], f.shape[2], f.shape[3]), dtype=f.dtype)
            f = torch.cat([f, pad], dim=0)
        elif f.shape[0] > max_len:
            f = f[:max_len]
        processed_frames.append(f)

    frames = torch.stack(processed_frames)
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


def create_data_loaders(train_subjects=['s1'], val_subjects=['s2'], test_subjects=['s3'],
                        max_samples_per_subject=None):
    """Create train, validation, and test data loaders.

    max_samples_per_subject caps utterances per speaker (e.g. for a quick local run).
    """

    def make(subjects, shuffle):
        dataset = GRIDDataset(Config.DATA_DIR, subjects=subjects,
                              max_samples_per_subject=max_samples_per_subject)
        return DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=shuffle,
                          collate_fn=collate_fn, num_workers=Config.NUM_WORKERS)

    train_loader = make(train_subjects, shuffle=True)
    val_loader = make(val_subjects, shuffle=False)
    test_loader = make(test_subjects, shuffle=False)

    return train_loader, val_loader, test_loader
