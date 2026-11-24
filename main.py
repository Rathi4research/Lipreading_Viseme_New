# Import project modules
import os

# On Apple Silicon (MPS), let ops that have no Metal kernel fall back to CPU instead of
# erroring. Must be set before torch is imported; no effect on CUDA/CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from src.config import Config, detect_device
import logging
import random
import subprocess
import numpy as np
import torch

import sys
import argparse
from src.data_loader import GRIDDataset, create_data_loaders, make_loader
from src.decoding import word_errors
from src.trainer import LipReadingTrainer
from src.evaluator import LipReadingEvaluator
from src.viseme_mapping import viseme_mapper
from models.fusion_network import ModelFactory

# Set up logging. force=True: imported modules (e.g. src.trainer) may already have configured the
# root logger, which would otherwise make this a silent no-op and leave lipreading.log empty.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lipreading.log'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories"""
    directories = ['checkpoints', 'logs', 'results', 'plots']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
    logger.info("Project directories created/verified")


def print_model_info(model):
    """Print model architecture information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("=" * 60)
    logger.info("MODEL ARCHITECTURE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size: ~{total_params * 4 / (1024 ** 2):.2f} MB")

    # Print model structure
    logger.info("\nModel Components:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        logger.info(f"  {name}: {params:,} parameters")
    logger.info("=" * 60)


def collapse_phonemes(indices):
    """Per-frame phoneme ids -> readable phoneme string (merge repeats, drop silence)"""
    out = []
    for idx in indices:
        p = Config.PHONEMES[idx] if 0 <= idx < len(Config.PHONEMES) else '?'
        if not out or out[-1] != p:
            out.append(p)
    return ' '.join(p for p in out if p != 'sil')


def resolve_checkpoint_dir(args):
    """Pick the checkpoint directory: --local-checkpoints > --checkpoint-dir > Config default."""
    if args.local_checkpoints:
        checkpoint_dir = Config.LOCAL_CHECKPOINT_DIR
    elif args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        checkpoint_dir = Config.CHECKPOINT_DIR

    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
    except OSError as e:
        raise SystemExit(
            f"Cannot create checkpoint directory '{checkpoint_dir}' ({e}). "
            f"The default is the Google Drive path used on Colab; when running locally pass "
            f"--local-checkpoints (saves to ./{Config.LOCAL_CHECKPOINT_DIR}) or --checkpoint-dir PATH."
        )
    return checkpoint_dir


def run_info(args):
    """Code version and command-line settings, stored in checkpoints and test results."""
    info = {'args': {k: v for k, v in vars(args).items()}}
    try:
        info['commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                        check=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], capture_output=True,
                               text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
        info['dirty'] = bool(dirty)
    except (OSError, subprocess.CalledProcessError):
        info['commit'] = None
    return info


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(args):
    return ModelFactory.create(args.model_type or 'full', seq_model=args.seq_model, visual_pool=args.visual_pool,
                               gate=args.gate, lm_context=args.lm_context)


def train_model(args):
    """Train the model"""
    logger.info("Starting model training...")
    checkpoint_dir = resolve_checkpoint_dir(args)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_subjects=args.train_subjects,
        val_subjects=args.val_subjects,
        test_subjects=args.test_subjects,
        max_samples_per_subject=args.max_samples_per_subject
    )

    # The model configuration is stored in the checkpoint and reused by evaluate/inference
    model = build_model(args)
    print_model_info(model)

    trainer = LipReadingTrainer(model, train_loader, val_loader, test_loader, checkpoint_dir=checkpoint_dir,
                                num_epochs=args.num_epochs, run_info=run_info(args))
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
        logger.info(f"Resumed training from {args.resume_from}")

    trainer.train(num_epochs=args.num_epochs)
    logger.info("Training completed!")


def evaluate_model(args):
    """Evaluate a checkpoint on the test speakers (optionally with controlled frame corruption)."""
    logger.info("Starting model evaluation...")
    if not args.checkpoint_path:
        logger.error("No checkpoint path provided for evaluation")
        return
    model = ModelFactory.load_pretrained_model(args.checkpoint_path, args.model_type)
    logger.info(f"Loaded model {model.model_config} from {args.checkpoint_path}")

    test_set = GRIDDataset(Config.DATA_DIR, subjects=args.test_subjects,
                           max_samples_per_subject=args.max_samples_per_subject, corrupt_ratio=args.corrupt_ratio)
    if args.corrupt_ratio:
        logger.info(f"Controlled corruption: about {args.corrupt_ratio:.0%} of each clip's frames replaced and flagged")
    if args.decode_head == 'draft' or args.drop_context:
        if getattr(model, 'draft_ctc_head', None) is None:
            raise ValueError("--decode-head draft / --drop-context need a model trained with --lm-context draft")
        model.eval_decode_head = args.decode_head
        model.eval_drop_context = args.drop_context
        logger.info(f"Evaluation variant: decode head '{args.decode_head}', draft context "
                    f"{'dropped (zeros)' if args.drop_context else 'on'}")
    evaluator = LipReadingEvaluator(model, make_loader(test_set, shuffle=False), device=Config.DEVICE)
    evaluator.evaluate_comprehensive()

    suffix = f"_corrupt{args.corrupt_ratio:g}" if args.corrupt_ratio else ""
    suffix += ("_drafthead" if args.decode_head == 'draft' else "") + ("_nocontext" if args.drop_context else "")
    out_dir = args.results_dir
    evaluator.save_plots(os.path.join(out_dir, 'plots'))
    evaluator.save_json(os.path.join(out_dir, f'evaluation_results{suffix}.json'),
                        extra={'checkpoint': args.checkpoint_path, 'test_subjects': args.test_subjects,
                               'corrupt_ratio': args.corrupt_ratio})
    evaluator.generate_report(os.path.join(out_dir, f'evaluation_report{suffix}.txt'))
    logger.info("Evaluation completed!")


def inference_demo(args):
    """Inference demo: decoded sentences vs references for a few test utterances"""
    logger.info("Running inference demo...")
    if not args.checkpoint_path:
        logger.error("No checkpoint path provided for inference")
        return
    model = ModelFactory.load_pretrained_model(args.checkpoint_path, args.model_type).to(Config.DEVICE)
    logger.info(f"Loaded model {model.model_config} from {args.checkpoint_path}")

    _, _, test_loader = create_data_loaders(test_subjects=args.test_subjects,
                                            max_samples_per_subject=args.max_samples_per_subject)
    shown = 0
    for batch in test_loader:
        if shown >= args.demo_samples:
            break
        speaker_ids = model.speaker_ids_for(batch['subjects'], Config.DEVICE)
        p = model.predict(batch['frames'].to(Config.DEVICE), lengths=batch['lengths'].to(Config.DEVICE),
                          speaker_ids=speaker_ids, quality=batch['quality'].to(Config.DEVICE))
        fw = None if p['fusion_weights'] is None else torch.stack(p['fusion_weights']).mean(0)
        for i in range(batch['frames'].size(0)):
            if shown >= args.demo_samples:
                break
            shown += 1
            ref = batch['sentences'][i]
            valid = batch['phonemes'][i] != -1
            logger.info(f"\n{'=' * 50}")
            logger.info(f"DEMO SAMPLE {shown}: subject {batch['subjects'][i]}, utterance {batch['utterance_ids'][i]}")
            logger.info(f"Reference:   {ref}")
            logger.info(f"Greedy:      {p['greedy_text'][i]}   (word errors {word_errors(ref, p['greedy_text'][i])[0]}/6)")
            logger.info(f"Constrained: {p['constrained_text'][i]}   (word errors {word_errors(ref, p['constrained_text'][i])[0]}/6)")
            p_pred = p['phoneme_predictions'][i].cpu()[valid].tolist()
            logger.info(f"Frame phonemes (pred):   {collapse_phonemes(p_pred)}")
            logger.info(f"Frame phonemes (target): {collapse_phonemes(batch['phonemes'][i][valid].tolist())}")
            if fw is not None:
                w = fw[i].cpu()[valid]
                logger.info(f"Fusion Weights - Visual: {w[:, 0].mean():.4f}, Language: {w[:, 1].mean():.4f}")
    logger.info("Inference demo completed!")


def analyze_dataset(args):
    """Analyze the dataset"""
    logger.info("Analyzing dataset...")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_subjects=args.train_subjects,
        val_subjects=args.val_subjects,
        test_subjects=args.test_subjects,
        max_samples_per_subject=args.max_samples_per_subject
    )

    # Dataset statistics
    total_samples = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)
    test_samples = len(test_loader.dataset)

    logger.info(f"Dataset Statistics:")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Training samples: {train_samples} ({train_samples / total_samples * 100:.1f}%)")
    logger.info(f"Validation samples: {val_samples} ({val_samples / total_samples * 100:.1f}%)")
    logger.info(f"Test samples: {test_samples} ({test_samples / total_samples * 100:.1f}%)")

    # Analyze sample batch
    sample_batch = next(iter(train_loader))
    frames = sample_batch['frames']
    visemes = sample_batch['visemes']
    phonemes = sample_batch['phonemes']
    ambiguity_scores = sample_batch['ambiguity_scores']

    logger.info(f"\nSample Batch Analysis:")
    logger.info(f"Frame shape: {frames.shape}")
    logger.info(f"Viseme sequence shape: {visemes.shape}")
    logger.info(f"Phoneme sequence shape: {phonemes.shape}")
    logger.info(f"Ambiguity scores shape: {ambiguity_scores.shape}")

    # Ambiguity distribution (ground truth, over non-padding frames)
    valid_ambiguity = ambiguity_scores[visemes != -1]
    logger.info(f"Ambiguity score range: {valid_ambiguity.min():.3f} - {valid_ambiguity.max():.3f}")
    logger.info(f"Average ambiguity: {valid_ambiguity.mean():.3f}")

    # Viseme distribution
    valid_visemes = visemes[visemes != -1]  # Remove padding
    unique_visemes, counts = torch.unique(valid_visemes, return_counts=True)

    logger.info(f"\nViseme Distribution:")
    for viseme_id, count in zip(unique_visemes.tolist(), counts.tolist()):
        percentage = count / len(valid_visemes) * 100
        phonemes_in_viseme = viseme_mapper.viseme_to_phonemes.get(viseme_id, [])
        logger.info(f"Viseme {viseme_id}: {count} samples ({percentage:.1f}%) - {phonemes_in_viseme}")

    # Print viseme mappings
    logger.info(f"\nViseme Mappings Summary:")
    viseme_mapper.print_mappings()


def create_sample_data(num_utterances=6, subjects=('s1', 's2', 's3')):
    """Create a tiny synthetic dataset for smoke-testing the pipeline (not real data).

    Writes GRID-format .align files (random grammatical GRID sentences, times in 1/25000 s) and
    random-noise lip-crop .npy files under Config.DATA_DIR, so train/evaluate/inference can run
    end to end without the real corpus. Accuracy numbers on this data are meaningless.
    """
    logger.info(f"Creating synthetic sample data under {Config.DATA_DIR} ...")
    rng = np.random.default_rng(0)
    slots = [Config.COMMANDS, Config.COLORS, Config.PREPOSITIONS, Config.LETTERS, Config.DIGITS, Config.ADVERBS]

    for subject in subjects:
        align_dir = os.path.join(Config.ALIGNMENT_DIR, subject)
        npy_dir = os.path.join(Config.PREPROCESSED_DIR, subject)
        os.makedirs(align_dir, exist_ok=True)
        os.makedirs(npy_dir, exist_ok=True)

        for n in range(num_utterances):
            words = [str(rng.choice(slot)) for slot in slots]
            utterance_id = f"sample{n:02d}"
            t, lines = 0, []
            for label, dur in [('sil', 12000)] + [(w, 9000) for w in words]:
                lines.append(f"{t} {t + dur} {label}")
                t += dur
            lines.append(f"{t} 74500 sil")
            with open(os.path.join(align_dir, utterance_id + '.align'), 'w') as f:
                f.write('\n'.join(lines) + '\n')

            num_frames = Config.SEQUENCE_LENGTH - (n % 2) * 3  # some clips slightly short
            size = Config.PREPROCESS_SIZE
            frames = rng.integers(0, 256, (num_frames, size, size), dtype=np.uint8)
            np.save(os.path.join(npy_dir, utterance_id + '.npy'), frames)
            # metadata like scripts/preprocess.py writes: a few missed detections and slight crop drift
            detected = np.ones(num_frames, dtype=bool)
            detected[rng.integers(0, num_frames, size=3)] = False
            transforms = np.tile(np.array([[1.0, 0, 0], [0, 1.0, 0]], dtype=np.float32), (num_frames, 1, 1))
            transforms[:, :, 2] = np.cumsum(rng.normal(0, 0.5, (num_frames, 2)), axis=0)
            np.savez_compressed(os.path.join(npy_dir, utterance_id + '.meta.npz'), detected=detected,
                                transforms=transforms, size=np.array(size), crop_mode=np.array('synthetic'))

    logger.info(f"Synthetic data created for {list(subjects)} ({num_utterances} utterances each).")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enhanced Viseme-Phoneme Adaptive Model for Lip Reading")

    # Mode selection
    parser.add_argument('mode', choices=['train', 'evaluate', 'inference', 'analyze', 'create_sample'],
                        help='Operation mode')

    # Model configuration
    parser.add_argument('--seq-model', choices=['bilstm', 'transformer', 'hybrid', 'conformer'],
                        default=Config.SEQ_MODEL, help='Temporal sequence encoder (training)')
    parser.add_argument('--visual-pool', choices=['avg', 'flatten'], default=Config.VISUAL_POOL,
                        help='Spatial pooling at the end of the 3D CNN (training)')
    parser.add_argument('--gate', choices=['ambiguity', 'entropy', 'learned', 'hard', 'intrinsic-extrinsic'],
                        default=Config.GATE, help='Fusion gate for --model-type full (training)')
    parser.add_argument('--lm-context', choices=['none', 'draft'], default=Config.LM_CONTEXT,
                        help="Language-model context: 'draft' conditions the language model on the "
                             "model's own draft CTC decode of the sequence features (corruption-"
                             "resistant sentence context; training)")
    parser.add_argument('--model-type', choices=['full', 'baseline', 'visual_only'],
                        default=None,
                        help="Model variant for training (default: full). For evaluate/inference "
                             "the type stored in the checkpoint is used.")
    parser.add_argument('--checkpoint-path', type=str, help='Path to model checkpoint')
    parser.add_argument('--resume-from', type=str, help='Resume training from checkpoint')

    # Data configuration
    parser.add_argument('--train-subjects', nargs='+', default=['s1'],
                        help='Subjects for training')
    parser.add_argument('--val-subjects', nargs='+', default=['s2'],
                        help='Subjects for validation')
    parser.add_argument('--test-subjects', nargs='+', default=['s3'],
                        help='Subjects for testing')

    # Training configuration
    parser.add_argument('--num-epochs', type=int, default=Config.NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=Config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='DataLoader worker processes (default: 2 on CUDA, 0 otherwise)')
    parser.add_argument('--seed', type=int, default=Config.SEED, help='Random seed')
    parser.add_argument('--select-metric', choices=['cer', 'wer'], default=Config.SELECT_METRIC,
                        help='Validation metric for the best checkpoint and early stopping')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision on CUDA')
    parser.add_argument('--min-detected-ratio', type=float, default=Config.MIN_DETECTED_RATIO,
                        help='Skip clips with a face detected in fewer than this share of frames')
    parser.add_argument('--corrupt-ratio', type=float, default=0.0,
                        help='evaluate: replace about this share of each test clip\'s frames and flag them '
                             '(controlled robustness test)')
    parser.add_argument('--decode-head', choices=['main', 'draft'], default='main',
                        help="evaluate (--lm-context draft models): decode the main CTC head or the interim draft head")
    parser.add_argument('--drop-context', action='store_true',
                        help='evaluate (--lm-context draft models): replace the draft context with zeros')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='evaluate: folder for the report, JSON and plots')

    # Checkpoint location
    ckpt_group = parser.add_mutually_exclusive_group()
    ckpt_group.add_argument('--local-checkpoints', action='store_true',
                            help=f'Save checkpoints/metrics to ./{Config.LOCAL_CHECKPOINT_DIR} '
                                 f'instead of the Google Drive default ({Config.CHECKPOINT_DIR})')
    ckpt_group.add_argument('--checkpoint-dir', type=str,
                            help='Save checkpoints/metrics to this directory')

    # Data location / size
    parser.add_argument('--data-dir', type=str, default=Config.DATA_DIR,
                        help='Root holding videos/, alignments/ and preprocessed_lips/')
    parser.add_argument('--frame-size', type=int,
                        help=f'Square model input size (default {Config.FRAME_HEIGHT}). Stored crops '
                             f'larger than this are random-cropped (train) / centre-cropped (eval); '
                             f'smaller ones are resized (e.g. pass 72 or 80 for older 80x80 .npy files)')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable training-time augmentation (random crop, flip, time masking)')
    parser.add_argument('--max-samples-per-subject', type=int,
                        help='Use at most N utterances per speaker (quick/local runs)')

    # Inference configuration
    parser.add_argument('--demo-samples', type=int, default=5,
                        help='Number of samples for demo')

    # Device configuration
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/mps/cpu/auto; auto picks cuda > mps > cpu)')

    args = parser.parse_args()

    # Setup
    setup_directories()

    # Set device
    if args.device == 'auto':
        device = detect_device()
    else:
        device = args.device
    Config.DEVICE = device

    # Update config with command line arguments (must happen before any model/dataset is built)
    Config.BATCH_SIZE = args.batch_size
    Config.LEARNING_RATE = args.learning_rate
    Config.NUM_WORKERS = args.num_workers if args.num_workers is not None else (2 if device.startswith('cuda') else 0)
    Config.MIN_DETECTED_RATIO = args.min_detected_ratio
    Config.AMP = not args.no_amp
    Config.SELECT_METRIC = args.select_metric
    Config.LM_CONTEXT = args.lm_context
    set_seed(args.seed)
    Config.set_data_dir(args.data_dir)
    if args.frame_size:
        Config.FRAME_WIDTH = Config.FRAME_HEIGHT = args.frame_size
    if args.no_augment:
        Config.AUGMENT = False

    logger.info(f"Using device: {Config.DEVICE}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model type: {args.model_type or ('full' if args.mode == 'train' else 'from checkpoint')}")
    logger.info(f"Data dir: {Config.DATA_DIR} | Model input: {Config.FRAME_HEIGHT}x{Config.FRAME_WIDTH} | "
                f"Train augmentation: {'on' if Config.AUGMENT else 'off'}")

    # Execute based on mode
    try:
        if args.mode == 'train':
            train_model(args)
        elif args.mode == 'evaluate':
            evaluate_model(args)
        elif args.mode == 'inference':
            inference_demo(args)
        elif args.mode == 'analyze':
            analyze_dataset(args)
        elif args.mode == 'create_sample':
            create_sample_data()

    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    except Exception as e:
        logger.error(f"Error during {args.mode}: {str(e)}")
        raise

    logger.info(f"{args.mode.capitalize()} completed successfully!")


if __name__ == "__main__":
    main()
