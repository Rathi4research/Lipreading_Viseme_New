import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.config import Config
from src.decoding import sentence_from_words, text_to_indices
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


def prepare_frames(frames, augment=False, return_mask=False):
    """Stored crops -> normalised model input (T, 1, FRAME_HEIGHT, FRAME_WIDTH) float tensor.

    Shared by training, evaluation and test.py so every path sees identical preprocessing.
      1. uint8 [0, 255] (or legacy float [0, 1]) -> float [0, 1]
      2. spatial: crops larger than the model input are cropped to it — at a random position when
         augmenting (training), centred otherwise; smaller/other sizes are resized
      3. normalise: (x - Config.PIXEL_MEAN) / Config.PIXEL_STD
      4. augment (training only, fresh per call): horizontal flip, time masking
    With return_mask=True also returns a (T,) bool tensor marking the time-masked frames.
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

    masked = torch.zeros(x.shape[0], dtype=torch.bool)
    if augment:
        if torch.rand(1).item() < Config.AUG_FLIP_PROB:
            x = torch.flip(x, dims=[-1])
        x, masked = time_mask(x)
    x = x.contiguous()
    return (x, masked) if return_mask else x


def time_mask(x, num_masks=None, max_len=None, generator=None):
    """Replace a few random short spans of frames with the clip's mean frame.

    Number of masks = AUG_TIME_MASKS_PER_SECOND x duration (TARGET_FPS frames per second), each up
    to AUG_TIME_MASK_MAX_FRAMES long (the "adaptive time masking" of the standard VSR recipes).
    Labels are left unchanged, so the model has to fill the gap from context.
    Returns (masked frames, (T,) bool mask of the replaced frames).
    """
    num_frames = x.shape[0]
    if num_masks is None:
        num_masks = int(round(Config.AUG_TIME_MASKS_PER_SECOND * num_frames / Config.TARGET_FPS))
    max_len = min(Config.AUG_TIME_MASK_MAX_FRAMES if max_len is None else max_len, num_frames)
    masked = torch.zeros(num_frames, dtype=torch.bool)
    if num_masks <= 0 or max_len <= 0:
        return x, masked
    x = x.clone()
    mean_frame = x.mean(dim=0, keepdim=True)
    for _ in range(num_masks):
        length = int(torch.randint(0, max_len + 1, (1,), generator=generator))
        if length == 0:
            continue
        start = int(torch.randint(0, num_frames - length + 1, (1,), generator=generator))
        x[start:start + length] = mean_frame
        masked[start:start + length] = True
    return x, masked


def meta_path_for(npy_path):
    return npy_path[:-len('.npy')] + '.meta.npz' if npy_path else None


def frame_quality(meta, num_frames):
    """Per-frame unreliability signals from a preprocessing .meta.npz (0 = reliable).

    Returns (quality (T, 3) float32, detected (T,) bool):
      [:, 0] missing  - 1 where dlib found no face (landmarks were interpolated)
      [:, 1] gap      - distance to the nearest detected frame / QUALITY_GAP_FRAMES, capped at 1
      [:, 2] jitter   - jump of the crop centre since the previous frame, in crop pixels /
                        QUALITY_JITTER_PX, capped at 1
    With no metadata every frame is treated as reliable.
    """
    quality = np.zeros((num_frames, Config.NUM_QUALITY_FEATURES), dtype=np.float32)
    detected = np.ones(num_frames, dtype=bool)
    if meta is None:
        return quality, detected

    det = np.asarray(meta['detected'], dtype=bool)[:num_frames]
    n = len(det)
    detected[:n] = det
    quality[:n, 0] = ~det
    if det.any():
        idx = np.flatnonzero(det)
        dist = np.abs(np.arange(n)[:, None] - idx[None, :]).min(axis=1)
        quality[:n, 1] = np.minimum(dist / Config.QUALITY_GAP_FRAMES, 1.0)
    else:
        quality[:n, 1] = 1.0

    tr = np.asarray(meta['transforms'], dtype=np.float64)[:n]
    if n > 1 and np.isfinite(tr).all():
        size = float(meta['size']) if 'size' in meta else Config.PREPROCESS_SIZE
        centre = np.array([size / 2.0, size / 2.0])
        # crop centre in original-frame pixels, and crop pixels per original pixel
        centres = np.stack([np.linalg.solve(m[:, :2], centre - m[:, 2]) for m in tr])
        scale = np.hypot(tr[:, 0, 0], tr[:, 0, 1])
        jump = np.linalg.norm(np.diff(centres, axis=0), axis=1) * scale[1:]
        quality[1:n, 2] = np.minimum(jump / Config.QUALITY_JITTER_PX, 1.0)
    return quality, detected


class GRIDDataset(Dataset):
    """GRID utterances as (frames, sentence target, per-frame labels, frame-quality signals).

    Frames come from the preprocessed lip crops in Config.PREPROCESSED_DIR
    (<subject>/<utterance_id>.npy, produced by scripts/preprocess.py), with the per-frame
    metadata in the matching .meta.npz when present. If an utterance has no .npy file, the raw
    video in Config.VIDEO_DIR is cropped on the fly with dlib instead (slow). Clips in which a
    face was detected in fewer than Config.MIN_DETECTED_RATIO of the frames are skipped.

    corrupt_ratio (evaluation only): deterministically replaces roughly that share of each clip's
    frames with the clip's mean frame (time-mask style) and flags them as unreliable — a
    controlled test of robustness to bad frames.
    """

    def __init__(self, data_dir, subjects=['s1', 's2', 's3'], transform=None, max_samples_per_subject=None,
                 augment=False, corrupt_ratio=0.0, min_detected_ratio=None):
        self.data_dir = data_dir
        self.subjects = subjects
        self.transform = transform
        # Training-time augmentation (random crop, flip, time masking); keep False for val/test
        self.augment = augment
        self.corrupt_ratio = corrupt_ratio
        min_detected_ratio = Config.MIN_DETECTED_RATIO if min_detected_ratio is None else min_detected_ratio
        self.samples = []
        self.skipped = []

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
                npy_path = os.path.join(npy_dir, utterance_id + '.npy') if utterance_id in npys else None
                meta_path = meta_path_for(npy_path)
                if meta_path and os.path.exists(meta_path):
                    with np.load(meta_path) as meta:
                        ratio = float(np.mean(meta['detected']))
                    if ratio < min_detected_ratio:
                        self.skipped.append((subject, utterance_id, ratio))
                        continue
                else:
                    meta_path = None
                subject_samples.append({
                    'video_path': videos.get(utterance_id),
                    'npy_path': npy_path,
                    'meta_path': meta_path,
                    'align_path': align_path,
                    'subject': subject,
                    'utterance_id': utterance_id
                })

            if max_samples_per_subject is not None:
                subject_samples = subject_samples[:max_samples_per_subject]
            self.samples.extend(subject_samples)

        skipped = f", skipped {len(self.skipped)} with face detected in < {min_detected_ratio:.0%} of frames" \
            if self.skipped else ""
        print(f"Loaded {len(self.samples)} samples from subjects {subjects}{skipped}")

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
        T = Config.SEQUENCE_LENGTH

        # Truncate to SEQUENCE_LENGTH, then crop/normalise/augment the real frames only
        frames = self._load_frames(sample)[:T]
        num_valid_frames = len(frames)
        video_tensor, masked = prepare_frames(frames, augment=self.augment, return_mask=True)

        meta = None
        if sample.get('meta_path'):
            with np.load(sample['meta_path']) as m:
                meta = {k: m[k] for k in ('detected', 'transforms', 'size')}
        quality, detected = frame_quality(meta, num_valid_frames)
        quality = torch.from_numpy(quality)
        detected = torch.from_numpy(detected)

        if self.corrupt_ratio > 0:
            g = torch.Generator().manual_seed(idx)
            num_masks = max(1, int(round(self.corrupt_ratio * num_valid_frames / 8)))
            video_tensor, corrupted = time_mask(video_tensor, num_masks=num_masks, max_len=12, generator=g)
            masked = masked | corrupted

        # Frames replaced by time masking carry no visual evidence: flag them as unreliable
        # (the extrinsic part of the fusion gate learns from this signal)
        quality[masked, 0] = 1.0

        # Pad to SEQUENCE_LENGTH with zeros (= the mean pixel after normalisation); these frames
        # get label -1 below and are ignored by the losses and metrics
        if num_valid_frames < T:
            padding = torch.zeros((T - num_valid_frames, *video_tensor.shape[1:]))
            video_tensor = torch.cat((video_tensor, padding), dim=0)
            quality = torch.cat((quality, torch.zeros(T - num_valid_frames, quality.shape[1])), dim=0)
            detected = torch.cat((detected, torch.zeros(T - num_valid_frames, dtype=torch.bool)), dim=0)

        # Per-frame labels (phonemes, visemes, words); padding frames are -1
        phonemes, words, visemes, word_labels = self.load_alignment(align_path, num_valid_frames)

        # Sentence-level CTC target (characters of the spoken words)
        sentence = sentence_from_words(words)

        # Ground-truth ambiguity scores — for analysis/metrics only, never a model input
        ambiguity_scores = [viseme_mapper.get_ambiguity_score(v) if v >= 0 else 0.0 for v in visemes]

        if self.transform:
            video_tensor = self.transform(video_tensor)

        return {
            'frames': video_tensor,
            'length': num_valid_frames,
            'phonemes': torch.LongTensor(phonemes),
            'visemes': torch.LongTensor(visemes),
            'word_labels': torch.LongTensor(word_labels),
            'ambiguity_scores': torch.FloatTensor(ambiguity_scores),
            'quality': quality,
            'detected': detected,
            'sentence': sentence,
            'target': torch.LongTensor(text_to_indices(sentence)),
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
    """Stack a batch; CTC targets are padded with 0 and come with their lengths."""
    targets = [item['target'] for item in batch]
    padded_targets = torch.zeros(len(batch), max(len(t) for t in targets), dtype=torch.long)
    for i, t in enumerate(targets):
        padded_targets[i, :len(t)] = t

    return {
        'frames': torch.stack([item['frames'] for item in batch]),
        'lengths': torch.tensor([item['length'] for item in batch], dtype=torch.long),
        'phonemes': torch.stack([item['phonemes'] for item in batch]),
        'visemes': torch.stack([item['visemes'] for item in batch]),
        'word_labels': torch.stack([item['word_labels'] for item in batch]),
        'ambiguity_scores': torch.stack([item['ambiguity_scores'] for item in batch]),
        'quality': torch.stack([item['quality'] for item in batch]),
        'detected': torch.stack([item['detected'] for item in batch]),
        'targets': padded_targets,
        'target_lengths': torch.tensor([len(t) for t in targets], dtype=torch.long),
        'sentences': [item['sentence'] for item in batch],
        'words': [item['words'] for item in batch],
        'subjects': [item['subject'] for item in batch],
        'utterance_ids': [item['utterance_id'] for item in batch]
    }


def make_loader(dataset, shuffle):
    """DataLoader with the project's collate function and device-appropriate loading options."""
    workers = Config.NUM_WORKERS
    return DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=shuffle, collate_fn=collate_fn,
                      num_workers=workers, pin_memory=str(Config.DEVICE).startswith('cuda'),
                      persistent_workers=workers > 0)


def create_data_loaders(train_subjects=['s1'], val_subjects=['s2'], test_subjects=['s3'],
                        max_samples_per_subject=None, test_corrupt_ratio=0.0):
    """Create train, validation, and test data loaders.

    max_samples_per_subject caps utterances per speaker (e.g. for a quick local run).
    test_corrupt_ratio applies the controlled frame corruption to the test split only.
    """

    def make(subjects, shuffle, augment, corrupt=0.0):
        dataset = GRIDDataset(Config.DATA_DIR, subjects=subjects, max_samples_per_subject=max_samples_per_subject,
                              augment=augment, corrupt_ratio=corrupt)
        return make_loader(dataset, shuffle)

    # Augmentation only ever applies to the training split (Config.AUGMENT, main.py --no-augment)
    train_loader = make(train_subjects, shuffle=True, augment=Config.AUGMENT)
    val_loader = make(val_subjects, shuffle=False, augment=False)
    test_loader = make(test_subjects, shuffle=False, augment=False, corrupt=test_corrupt_ratio)

    return train_loader, val_loader, test_loader
