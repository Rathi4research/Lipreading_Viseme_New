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


def prepare_frames(frames, augment=False):
    """Stored crops -> normalised model input (T, 1, FRAME_HEIGHT, FRAME_WIDTH) float tensor.

    Shared by training, evaluation and test.py so every path sees identical preprocessing.
      1. uint8 [0, 255] (or legacy float [0, 1]) -> float [0, 1]
      2. spatial: crops larger than the model input are cropped to it — at a random position when
         augmenting (training), centred otherwise; smaller/other sizes are resized
      3. normalise: (x - Config.PIXEL_MEAN) / Config.PIXEL_STD
      4. augment (training only, fresh per call): horizontal flip, time masking
    """
    x = torch.as_tensor(np.asarray(frames))
    x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
    if x.dim() == 3:
        x = x.unsqueeze(1)  # (T, 1, H, W)

    th, tw = Config.FRAME_HEIGHT, Config.FRAME_WIDTH
    h, w = x.shape[-2:]
    if h >= th and w >= tw and (h, w) != (th, tw):
        if augment:
            top = int(torch.randint(0, h - th + 1, (1,)))
            left = int(torch.randint(0, w - tw + 1, (1,)))
        else:
            top, left = (h - th) // 2, (w - tw) // 2
        x = x[..., top:top + th, left:left + tw]
    elif (h, w) != (th, tw):
        x = F.interpolate(x, size=(th, tw), mode='bilinear', align_corners=False)

    x = (x - Config.PIXEL_MEAN) / Config.PIXEL_STD

    if augment:
        if torch.rand(1).item() < Config.AUG_FLIP_PROB:
            x = torch.flip(x, dims=[-1])
        x = time_mask(x)
    return x.contiguous()


def time_mask(x):
    """Replace a few random short spans of frames with the clip's mean frame.

    Number of masks = AUG_TIME_MASKS_PER_SECOND x duration (TARGET_FPS frames per second), each up
    to AUG_TIME_MASK_MAX_FRAMES long (the "adaptive time masking" of the standard VSR recipes).
    Labels are left unchanged, so the model has to fill the gap from context.
    """
    num_frames = x.shape[0]
    num_masks = int(round(Config.AUG_TIME_MASKS_PER_SECOND * num_frames / Config.TARGET_FPS))
    max_len = min(Config.AUG_TIME_MASK_MAX_FRAMES, num_frames)
    if num_masks <= 0 or max_len <= 0:
        return x
    x = x.clone()
    mean_frame = x.mean(dim=0, keepdim=True)
    for _ in range(num_masks):
        length = int(torch.randint(0, max_len + 1, (1,)))
        if length == 0:
            continue
        start = int(torch.randint(0, num_frames - length + 1, (1,)))
        x[start:start + length] = mean_frame
    return x


class GRIDDataset(Dataset):
    """GRID utterances as (frames, per-frame phoneme/viseme labels).

    Frames come from the preprocessed lip crops in Config.PREPROCESSED_DIR
    (<subject>/<utterance_id>.npy, produced by scripts/preprocess.py). If an utterance has no
    .npy file, the raw video in Config.VIDEO_DIR is cropped on the fly with dlib instead (slow).
    Samples are discovered from either source, so a local run only needs the .npy files and
    the .align files — the raw videos are optional.
    """

    def __init__(self, data_dir, subjects=['s1', 's2', 's3'], transform=None, max_samples_per_subject=None,
                 augment=False):
        self.data_dir = data_dir
        self.subjects = subjects
        self.transform = transform
        # Training-time augmentation (random crop, flip, time masking); keep False for val/test
        self.augment = augment
        self.samples = []

        # dlib is only needed for the on-the-fly fallback; initialised lazily.
        self._dlib_ready = False
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
        """Return the stored crops as a numpy array (T, H, W): uint8, or float [0, 1] for legacy files."""
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

        return frames

    def __getitem__(self, idx):
        sample = self.samples[idx]
        align_path = sample['align_path']

        # Truncate to SEQUENCE_LENGTH, then crop/normalise/augment the real frames only
        frames = self._load_frames(sample)[:Config.SEQUENCE_LENGTH]
        num_valid_frames = len(frames)
        video_tensor = prepare_frames(frames, augment=self.augment)

        # Pad to SEQUENCE_LENGTH with zeros (= the mean pixel after normalisation); these frames
        # get label -1 below and are ignored by the losses and metrics
        if num_valid_frames < Config.SEQUENCE_LENGTH:
            padding = torch.zeros((Config.SEQUENCE_LENGTH - num_valid_frames, *video_tensor.shape[1:]))
            video_tensor = torch.cat((video_tensor, padding), dim=0)

        # Per-frame labels (phonemes, visemes, words); padding frames are -1
        phonemes, words, visemes, word_labels = self.load_alignment(align_path, num_valid_frames)

        # Convert to tensors
        phonemes_tensor = torch.LongTensor(phonemes)
        visemes_tensor = torch.LongTensor(visemes)
        word_labels_tensor = torch.LongTensor(word_labels)

        # Ground-truth ambiguity scores — for analysis/metrics only, never a model input
        ambiguity_scores = [viseme_mapper.get_ambiguity_score(v) for v in visemes]
        ambiguity_tensor = torch.FloatTensor(ambiguity_scores)

        # Apply transforms if provided
        if self.transform:
            video_tensor = self.transform(video_tensor)

        return {
            'frames': video_tensor,
            'phonemes': phonemes_tensor,
            'visemes': visemes_tensor,
            'word_labels': word_labels_tensor,
            'ambiguity_scores': ambiguity_tensor,
            'words': words,
            'subject': sample['subject'],
            'utterance_id': sample['utterance_id']
        }

    def load_video(self, video_path):
        """Crop the lip region from a raw video on the fly (fallback when no .npy exists).

        Uses the same cropping as scripts/preprocess.py so on-the-fly and preprocessed samples
        look alike. Needs dlib and shape_predictor_68_face_landmarks.dat; slow.
        Returns uint8 frames (T, PREPROCESS_SIZE, PREPROCESS_SIZE), like the stored .npy files.
        """
        from scripts import preprocess

        if not self._dlib_ready:
            if preprocess.dlib is None:
                raise RuntimeError("dlib is required to crop raw videos; run scripts/preprocess.py "
                                   "(e.g. on Colab) to create .npy files instead")
            preprocess._init_worker("shape_predictor_68_face_landmarks.dat")
            self._dlib_ready = True

        frames = preprocess.extract_lip_sequence(video_path, size=Config.PREPROCESS_SIZE,
                                                 max_frames=Config.SEQUENCE_LENGTH)
        if not frames:
            raise RuntimeError(f"Could not read any frames from {video_path}")
        return np.asarray(frames, dtype=np.uint8)

    def load_alignment(self, align_path, num_valid_frames=None):
        """Build per-frame phoneme/viseme/word label sequences aligned to the video's frame grid.

        Returns (phoneme_indices, words, viseme_indices, word_indices), the index lists being
        Config.SEQUENCE_LENGTH long; frames >= num_valid_frames (padding) are labelled -1.

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
        frame_words = ['sil'] * num_frames

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
                frame_words[t] = label if label in Config.VOCABULARY else 'sil'

        # Convert per-frame labels to indices
        phoneme_to_idx = {p: i for i, p in enumerate(Config.PHONEMES)}
        phoneme_indices = [phoneme_to_idx.get(p, 0) for p in frame_phonemes]
        viseme_indices = viseme_mapper.phoneme_to_viseme_sequence(frame_phonemes)
        word_to_idx = {w: i for i, w in enumerate(Config.VOCABULARY)}
        word_indices = [word_to_idx[w] for w in frame_words]

        # Frames past the end of the actual video are zero padding, not silence: label them -1
        # so they are ignored by the losses and accuracy metrics.
        if num_valid_frames is not None and num_valid_frames < num_frames:
            for t in range(num_valid_frames, num_frames):
                phoneme_indices[t] = viseme_indices[t] = word_indices[t] = -1

        return phoneme_indices, words, viseme_indices, word_indices


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
    word_labels = torch.stack([item['word_labels'] for item in batch])
    ambiguity_scores = torch.stack([item['ambiguity_scores'] for item in batch])

    return {
        'frames': frames,
        'phonemes': phonemes,
        'visemes': visemes,
        'word_labels': word_labels,
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

    def make(subjects, shuffle, augment):
        dataset = GRIDDataset(Config.DATA_DIR, subjects=subjects,
                              max_samples_per_subject=max_samples_per_subject, augment=augment)
        return DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=shuffle,
                          collate_fn=collate_fn, num_workers=Config.NUM_WORKERS)

    # Augmentation only ever applies to the training split (Config.AUGMENT, main.py --no-augment)
    train_loader = make(train_subjects, shuffle=True, augment=Config.AUGMENT)
    val_loader = make(val_subjects, shuffle=False, augment=False)
    test_loader = make(test_subjects, shuffle=False, augment=False)

    return train_loader, val_loader, test_loader
