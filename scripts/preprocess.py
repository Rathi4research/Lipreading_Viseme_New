"""Pre-extract mouth-region crops from GRID videos.

Runs dlib face/landmark detection once per video (in parallel across CPU cores) instead of on
every epoch inside the Dataset. Output layout, read by src/data_loader.py:

    <output-dir>/<speaker>/<utterance_id>.npy        uint8 array (T, size, size), grayscale mouth crops
    <output-dir>/<speaker>/<utterance_id>.meta.npz   per-frame metadata (see below; --no-meta to skip)

Pipeline (crop_mode='aligned', the default — follows the standard modern VSR recipe):
  1. dlib 68-point landmarks on every frame.
  2. Frames where detection failed are filled by linear interpolation between the nearest
     detected frames (copied from the nearest detection at the clip edges).
  3. Landmarks are smoothed over time (centred moving average, --smooth-window frames) — used
     only to place the crop, never to alter the pixels, so lip motion itself is untouched.
  4. Per-frame similarity transform: rotate so the eye line is horizontal, scale so the crop
     side is --crop-scale x the distance between the eye centres (i.e. sized by the FACE, not
     by the lips, so the zoom does not change when the mouth opens or rounds), centred on the
     midpoint of the mouth corners. Applied with warpAffine to a size x size crop.
  5. Saved as uint8 (normalisation and augmentation happen in the data loader at train time,
     freshly per sample, and only for training data).

Other crop modes, kept for ablations / reproducing older files:
  - 'square': square box sized from the lip bounding box (zoom varies with mouth shape), smoothed.
  - 'tight' : per-frame lip bounding box (+pad) stretched to a square (the original notebook crop).

Metadata (.meta.npz), for later use (e.g. geometric / reliability features) without re-running dlib:
  landmarks      (T, 68, 2) float32  raw detections in original video pixels, NaN where missed
  detected       (T,)       bool     whether dlib found a face in that frame
  transforms     (T, 2, 3)  float32  affine matrix mapping original pixels -> crop pixels
  landmarks_crop (T, 68, 2) float32  interpolated (unsmoothed) landmarks mapped into crop pixels
  frame_shape    (2,)       int      original frame height, width
  crop_mode, size, crop_scale, smooth_window

Examples:
    python scripts/preprocess.py --speakers s1 s2 s3
    python scripts/preprocess.py --backup-dir /content/drive/MyDrive/Lipreading_Preprocessed_v2
"""
import argparse
import os
import sys
import tarfile
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    import dlib
except ImportError:  # reported in main() so --help still works without dlib
    dlib = None

DEFAULT_SIZE = 96
DEFAULT_CROP_SCALE = 1.4
DEFAULT_SMOOTH_WINDOW = 12

_detector = None
_predictor = None


def _init_worker(landmark_path):
    global _detector, _predictor
    _detector = dlib.get_frontal_face_detector()
    _predictor = dlib.shape_predictor(landmark_path)


# ---------------------------------------------------------------- landmark tracking
def _detect_landmarks(gray):
    """(68, 2) float array of dlib landmarks for the first detected face, or None"""
    faces = _detector(gray, 0)
    if len(faces) == 0:
        return None
    shape = _predictor(gray, faces[0])
    return np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.float64)


def interpolate_landmarks(raw):
    """Fill missed frames (NaN) by linear interpolation in time; edges copy the nearest detection.

    raw: (T, 68, 2) with NaN rows for missed frames. Returns a filled copy (all-NaN input stays NaN).
    """
    filled = raw.copy()
    detected = ~np.isnan(raw[:, 0, 0])
    if detected.all() or not detected.any():
        return filled
    t = np.arange(len(raw))
    flat = filled.reshape(len(raw), -1)
    for j in range(flat.shape[1]):
        flat[:, j] = np.interp(t, t[detected], flat[detected, j])  # np.interp clamps at the edges
    return flat.reshape(raw.shape)


def smooth_sequence(values, window):
    """Centred moving average along axis 0 (edge-padded) for an array of any trailing shape"""
    if window <= 1 or len(values) < 2:
        return values
    half_lo, half_hi = (window - 1) // 2, window // 2
    pad = [(half_lo, half_hi)] + [(0, 0)] * (values.ndim - 1)
    padded = np.pad(values, pad, mode='edge')
    cumsum = np.cumsum(padded, axis=0)
    cumsum = np.concatenate([np.zeros_like(cumsum[:1]), cumsum], axis=0)
    return (cumsum[window:] - cumsum[:-window]) / window


# ---------------------------------------------------------------- crop geometry
def aligned_transform(landmarks, size, crop_scale=DEFAULT_CROP_SCALE):
    """2x3 similarity transform (original pixels -> size x size crop) for one frame's landmarks.

    Rotation levels the line between the eye centres; scale makes the crop side equal to
    crop_scale x the eye-centre distance; the midpoint of the mouth corners lands at the crop centre.
    """
    left_eye = landmarks[36:42].mean(axis=0)
    right_eye = landmarks[42:48].mean(axis=0)
    dx, dy = right_eye - left_eye
    angle = np.degrees(np.arctan2(dy, dx))
    eye_dist = max(np.hypot(dx, dy), 1e-6)
    center = (landmarks[48] + landmarks[54]) / 2.0

    m = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), float(angle), size / (crop_scale * eye_dist))
    m[:, 2] += np.array([size / 2.0, size / 2.0]) - center
    return m


def square_transform(lip_box, size):
    """2x3 transform for an axis-aligned square crop: lip_box = (cx, cy, side)"""
    cx, cy, side = lip_box
    s = size / max(side, 1e-6)
    return np.array([[s, 0, size / 2.0 - s * cx], [0, s, size / 2.0 - s * cy]], dtype=np.float64)


def _lip_box(landmarks):
    lips = landmarks[48:68]
    x0, y0 = lips.min(axis=0)
    x1, y1 = lips.max(axis=0)
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0


def _apply_transform(points, m):
    return points @ m[:, :2].T + m[:, 2]


# ---------------------------------------------------------------- main extraction
def extract_lip_sequence(video_path, size=DEFAULT_SIZE, max_frames=75, pad_ratio=0.08,
                         crop_mode='aligned', smooth_window=DEFAULT_SMOOTH_WINDOW,
                         crop_scale=DEFAULT_CROP_SCALE, return_meta=False):
    """Crop the mouth region from each frame -> list of (size, size) uint8 frames.

    See the module docstring for the crop modes. With return_meta=True also returns the metadata
    dict that process_video() saves as .meta.npz.
    """
    cap = cv2.VideoCapture(str(video_path))
    grays, raw = [], []
    while len(grays) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grays.append(gray)
        lm = _detect_landmarks(gray)
        raw.append(lm if lm is not None else np.full((68, 2), np.nan))
    cap.release()

    if not grays:
        return ([], None) if return_meta else []

    raw = np.stack(raw)
    detected = ~np.isnan(raw[:, 0, 0])
    h, w = grays[0].shape

    if not detected.any():  # no face anywhere: black frames, identity-free metadata
        frames = [np.zeros((size, size), dtype=np.uint8) for _ in grays]
        transforms = np.full((len(grays), 2, 3), np.nan)
        filled = raw
    else:
        filled = interpolate_landmarks(raw)
        if crop_mode == 'aligned':
            smoothed = smooth_sequence(filled, smooth_window)
            transforms = np.stack([aligned_transform(lm, size, crop_scale) for lm in smoothed])
        elif crop_mode == 'square':
            boxes = np.array([_lip_box(lm) for lm in filled])  # cx, cy, w, h
            sides = boxes[:, 2:].max(axis=1, keepdims=True) * (1 + 2 * pad_ratio)
            track = smooth_sequence(np.hstack([boxes[:, :2], sides]), smooth_window)
            transforms = np.stack([square_transform(b, size) for b in track])
        elif crop_mode == 'tight':
            transforms = []
            for lm in filled:  # per-frame box, no smoothing, aspect ratio not preserved
                cx, cy, bw, bh = _lip_box(lm)
                pad = max(bw, bh) * pad_ratio
                sx, sy = size / max(bw + 2 * pad, 1e-6), size / max(bh + 2 * pad, 1e-6)
                transforms.append(np.array([[sx, 0, size / 2.0 - sx * cx], [0, sy, size / 2.0 - sy * cy]]))
            transforms = np.stack(transforms)
        else:
            raise ValueError(f"unknown crop_mode {crop_mode!r}")

        frames = [cv2.warpAffine(g, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                  for g, m in zip(grays, transforms)]

    if not return_meta:
        return frames

    landmarks_crop = np.stack([_apply_transform(lm, m) for lm, m in zip(filled, transforms)])
    meta = {
        'landmarks': raw.astype(np.float32),
        'detected': detected,
        'transforms': transforms.astype(np.float32),
        'landmarks_crop': landmarks_crop.astype(np.float32),
        'frame_shape': np.array([h, w]),
        'crop_mode': np.array(crop_mode),
        'size': np.array(size),
        'crop_scale': np.array(crop_scale),
        'smooth_window': np.array(smooth_window),
    }
    return frames, meta


def process_video(task):
    out_path = task['out_path']
    if os.path.exists(out_path):
        return 'skipped'
    try:
        frames, meta = extract_lip_sequence(task['video_path'], task['size'], task['max_frames'],
                                            task['pad_ratio'], task['crop_mode'], task['smooth_window'],
                                            task['crop_scale'], return_meta=True)
        if not frames:
            return f"error: no frames in {task['video_path']}"
        if task['save_meta']:
            np.savez_compressed(out_path[:-len('.npy')] + '.meta.npz', **meta)
        # Write frames last: the .npy's existence marks the utterance as done (resumable runs)
        np.save(out_path, np.asarray(frames, dtype=np.uint8))
        if not meta['detected'].all():
            return f"ok (no face in {int((~meta['detected']).sum())}/{len(frames)} frames)"
        return 'ok'
    except Exception as e:  # keep going; report at the end
        return f"error: {task['video_path']}: {e}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--video-dir', default='data/videos', help='<video-dir>/<speaker>/*.mpg')
    parser.add_argument('--output-dir', default='data/preprocessed_lips', help='Where crops are written')
    parser.add_argument('--speakers', nargs='+', help='Speakers to process (default: all found in --video-dir)')
    parser.add_argument('--size', type=int, default=DEFAULT_SIZE,
                        help=f'Stored crop size (square), default {DEFAULT_SIZE}; the loader random-crops '
                             f'the model input (default 88) from it during training')
    parser.add_argument('--max-frames', type=int, default=75, help='Max frames per video, default 75')
    parser.add_argument('--crop-mode', choices=['aligned', 'square', 'tight'], default='aligned',
                        help="'aligned' (default): face-aligned, face-scaled crop. 'square': lip-box-scaled "
                             "square crop. 'tight': per-frame lip box stretched to a square (original notebook).")
    parser.add_argument('--crop-scale', type=float, default=DEFAULT_CROP_SCALE,
                        help=f"'aligned' crop side as a multiple of the eye-centre distance (default {DEFAULT_CROP_SCALE})")
    parser.add_argument('--smooth-window', type=int, default=DEFAULT_SMOOTH_WINDOW,
                        help=f'Frames in the moving average that smooths the crop track (default {DEFAULT_SMOOTH_WINDOW})')
    parser.add_argument('--pad-ratio', type=float, default=0.08,
                        help="'square'/'tight' padding around the lip box as a fraction of its larger side")
    parser.add_argument('--no-meta', action='store_true', help='Do not write the .meta.npz landmark files')
    parser.add_argument('--workers', type=int, default=cpu_count(), help='Parallel processes')
    parser.add_argument('--landmarks', default='shape_predictor_68_face_landmarks.dat',
                        help='Path to dlib 68-point landmark model')
    parser.add_argument('--backup-dir',
                        help='If set, write <speaker>_preprocessed.tar.gz here after each speaker '
                             '(e.g. a Google Drive folder) and skip speakers already backed up')
    args = parser.parse_args()

    if dlib is None:
        sys.exit("dlib is required for preprocessing: pip install dlib "
                 "(preinstalled on Colab; on macOS needs cmake: brew install cmake)")
    if not os.path.exists(args.landmarks):
        sys.exit(f"Landmark model not found: {args.landmarks}\n"
                 "Download: wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && "
                 "bunzip2 shape_predictor_68_face_landmarks.dat.bz2")

    video_root, out_root = Path(args.video_dir), Path(args.output_dir)
    speakers = args.speakers or sorted(d.name for d in video_root.iterdir() if d.is_dir())
    if args.backup_dir:
        os.makedirs(args.backup_dir, exist_ok=True)

    print(f"Preprocessing {len(speakers)} speaker(s) with {args.workers} workers: size={args.size}, "
          f"crop={args.crop_mode} (scale {args.crop_scale}, smooth {args.smooth_window}), "
          f"max_frames={args.max_frames}, meta={not args.no_meta}")

    failures = []
    for speaker in speakers:
        archive = os.path.join(args.backup_dir, f"{speaker}_preprocessed.tar.gz") if args.backup_dir else None
        if archive and os.path.exists(archive):
            print(f"[{speaker}] backup already exists at {archive} - skipping")
            continue

        videos = sorted((video_root / speaker).glob('*.mpg')) + sorted((video_root / speaker).glob('*.mp4'))
        if not videos:
            print(f"[{speaker}] no videos found in {video_root / speaker} - skipping")
            continue

        out_dir = out_root / speaker
        out_dir.mkdir(parents=True, exist_ok=True)
        tasks = [{'video_path': str(v), 'out_path': str(out_dir / (v.stem + '.npy')), 'size': args.size,
                  'max_frames': args.max_frames, 'pad_ratio': args.pad_ratio, 'crop_mode': args.crop_mode,
                  'smooth_window': args.smooth_window, 'crop_scale': args.crop_scale,
                  'save_meta': not args.no_meta}
                 for v in videos]

        with Pool(processes=args.workers, initializer=_init_worker, initargs=(args.landmarks,)) as pool:
            results = list(tqdm(pool.imap_unordered(process_video, tasks), total=len(tasks), desc=speaker))

        errors = [r for r in results if r.startswith('error')]
        partial = [r for r in results if r.startswith('ok (')]
        failures.extend(errors)
        print(f"[{speaker}] ok={sum(r.startswith('ok') for r in results)} skipped={results.count('skipped')} "
              f"errors={len(errors)} clips_with_missed_faces={len(partial)}")

        if archive:
            with tarfile.open(archive, 'w:gz') as tar:
                tar.add(str(out_dir), arcname=speaker)
            print(f"[{speaker}] backup written to {archive}")

    if failures:
        print(f"\n{len(failures)} video(s) failed:")
        for f in failures[:20]:
            print("  " + f)
        sys.exit(1)
    print("\nPreprocessing complete.")


if __name__ == '__main__':
    main()
