"""Pre-extract lip-region crops from GRID videos into .npy files.

Runs dlib face/landmark detection once per video (in parallel across CPU cores) instead of
on every epoch inside the Dataset. Output layout, read by src/data_loader.py:

    <output-dir>/<speaker>/<utterance_id>.npy     float32 array, shape (T, size, size), values in [0, 1]

Ported from the Colab notebook's "Multi-Core Accelerated Preprocessing" cell.

Examples:
    python scripts/preprocess.py --speakers s1 s2 s3 --size 80
    python scripts/preprocess.py --size 80 --backup-dir /content/drive/MyDrive/Lipreading_Preprocessed_Backup
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

_detector = None
_predictor = None


def _init_worker(landmark_path):
    global _detector, _predictor
    _detector = dlib.get_frontal_face_detector()
    _predictor = dlib.shape_predictor(landmark_path)


def _augment(frames, flip=False, rotate_angle=0.0):
    """Same geometric transform for every frame so the sequence stays temporally consistent."""
    out = []
    for img in frames:
        img = img.copy()
        if rotate_angle != 0:
            h, w = img.shape[:2]
            m = cv2.getRotationMatrix2D((w // 2, h // 2), rotate_angle, 1.0)
            img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if flip:
            img = cv2.flip(img, 1)
        out.append(img)
    return out


def extract_lip_sequence(video_path, size, max_frames, pad_ratio):
    """Crop the lip bounding box (dlib landmarks 48-67, padded) from each frame and resize."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _detector(gray, 0)
        if len(faces) > 0:
            shape = _predictor(gray, faces[0])
            lip_pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)])
            x, y, w, h = cv2.boundingRect(lip_pts)
            pad = int(max(w, h) * pad_ratio)
            y1, y2 = max(0, y - pad), min(frame.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(frame.shape[1], x + w + pad)
            crop = gray[y1:y2, x1:x2]
            frames.append(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA))
        elif frames:
            frames.append(frames[-1])  # detector missed: repeat previous crop
        else:
            frames.append(np.zeros((size, size), dtype=np.uint8))
    cap.release()
    return frames


def process_video(task):
    video_path, out_path, size, max_frames, pad_ratio, augment, seed = task
    if os.path.exists(out_path):
        return 'skipped'
    try:
        frames = extract_lip_sequence(video_path, size, max_frames, pad_ratio)
        if not frames:
            return f'error: no frames in {video_path}'
        if augment:
            rng = np.random.default_rng(seed)
            frames = _augment(frames, flip=rng.random() > 0.5, rotate_angle=rng.uniform(-5, 5))
        np.save(out_path, np.asarray(frames, dtype=np.float32) / 255.0)
        return 'ok'
    except Exception as e:  # keep going; report at the end
        return f'error: {video_path}: {e}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--video-dir', default='data/videos', help='<video-dir>/<speaker>/*.mpg')
    parser.add_argument('--output-dir', default='data/preprocessed_lips', help='Where .npy crops are written')
    parser.add_argument('--speakers', nargs='+', help='Speakers to process (default: all found in --video-dir)')
    parser.add_argument('--size', type=int, default=80, help='Output crop size (square), default 80')
    parser.add_argument('--max-frames', type=int, default=75, help='Max frames per video, default 75')
    parser.add_argument('--pad-ratio', type=float, default=0.08,
                        help='Padding around the lip bounding box as a fraction of its larger side')
    parser.add_argument('--augment', action='store_true',
                        help='Apply one random flip/rotation per video (baked into the saved file, '
                             'so it also affects val/test speakers). Off by default.')
    parser.add_argument('--seed', type=int, default=0, help='Base seed for --augment')
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

    print(f"Preprocessing {len(speakers)} speaker(s) with {args.workers} workers: "
          f"size={args.size}, max_frames={args.max_frames}, augment={args.augment}")

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
        tasks = [(str(v), str(out_dir / (v.stem + '.npy')), args.size, args.max_frames,
                  args.pad_ratio, args.augment, args.seed + i)
                 for i, v in enumerate(videos)]

        with Pool(processes=args.workers, initializer=_init_worker, initargs=(args.landmarks,)) as pool:
            results = list(tqdm(pool.imap_unordered(process_video, tasks), total=len(tasks), desc=speaker))

        errors = [r for r in results if r.startswith('error')]
        failures.extend(errors)
        print(f"[{speaker}] ok={results.count('ok')} skipped={results.count('skipped')} errors={len(errors)}")

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
