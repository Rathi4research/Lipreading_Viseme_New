# Import project modules
import os

# On Apple Silicon (MPS), let ops that have no Metal kernel fall back to CPU instead of
# erroring. Must be set before torch is imported; no effect on CUDA/CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from src.config import Config, detect_device
import logging
import numpy as np
import torch

import sys
import argparse
from src.data_loader import create_data_loaders
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


def train_model(args):
    """Train the model"""
    logger.info("Starting model training...")

    checkpoint_dir = resolve_checkpoint_dir(args)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_subjects=args.train_subjects,
        val_subjects=args.val_subjects,
        test_subjects=args.test_subjects,
        max_samples_per_subject=args.max_samples_per_subject
    )

    # Create model (same model_type is used for training and, via the checkpoint, evaluation)
    model = ModelFactory.create(args.model_type or 'full')

    print_model_info(model)

    # Print viseme mappings
    logger.info("Viseme-Phoneme Mappings:")
    viseme_mapper.print_mappings()

    # Create trainer
    trainer = LipReadingTrainer(model, train_loader, val_loader, test_loader,
                                checkpoint_dir=checkpoint_dir)

    # Load checkpoint if specified
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
        logger.info(f"Resumed training from {args.resume_from}")

    # Start training
    trainer.train(num_epochs=args.num_epochs)

    logger.info("Training completed!")


def evaluate_model(args):
    """Evaluate the model"""
    logger.info("Starting model evaluation...")

    # Create test data loader
    _, _, test_loader = create_data_loaders(test_subjects=args.test_subjects,
                                            max_samples_per_subject=args.max_samples_per_subject)

    # Load model
    if args.checkpoint_path:
        model = ModelFactory.load_pretrained_model(args.checkpoint_path, args.model_type)
        logger.info(f"Loaded model from {args.checkpoint_path}")
    else:
        logger.error("No checkpoint path provided for evaluation")
        return

    # Create evaluator
    evaluator = LipReadingEvaluator(model, test_loader, device=Config.DEVICE)

    # Run comprehensive evaluation
    results = evaluator.evaluate_comprehensive()

    # Generate plots
    if 'confusion_matrices' in results:
        # Plot viseme confusion matrix (labels show the phonemes in each viseme class)
        viseme_labels = [f"V{i} " + "/".join(p for p in viseme_mapper.viseme_to_phonemes[i] if p in Config.PHONEMES)[:12]
                         for i in range(viseme_mapper.num_visemes)]
        evaluator.plot_confusion_matrix(
            results['confusion_matrices']['viseme_confusion_matrix'],
            viseme_labels,
            "Viseme Confusion Matrix",
            "plots/viseme_confusion_matrix.png"
        )

        # Plot phoneme confusion matrix restricted to phonemes that occur in the test set
        phoneme_cm = results['confusion_matrices']['phoneme_confusion_matrix']
        present = [i for i in range(phoneme_cm.shape[0]) if phoneme_cm[i].sum() > 0]
        evaluator.plot_confusion_matrix(
            phoneme_cm[np.ix_(present, present)],
            [Config.PHONEMES[i] for i in present],
            "Phoneme Confusion Matrix (phonemes present in test set)",
            "plots/phoneme_confusion_matrix.png",
            figsize=(max(12, len(present) * 0.45), max(10, len(present) * 0.4))
        )



    # Plot fusion analysis
    if 'fusion_analysis' in results:
        evaluator.plot_fusion_analysis(
            results['fusion_analysis'],
            "plots/fusion_analysis.png"
        )

    # Generate report
    evaluator.generate_report("results/evaluation_report.txt")

    logger.info("Evaluation completed!")


def inference_demo(args):
    """Run inference demo: per-utterance predictions vs targets for a few test samples"""
    logger.info("Running inference demo...")

    if not args.checkpoint_path:
        logger.error("No checkpoint path provided for inference")
        return
    model = ModelFactory.load_pretrained_model(args.checkpoint_path, args.model_type)
    logger.info(f"Loaded '{model.model_type}' model from {args.checkpoint_path}")

    _, _, test_loader = create_data_loaders(test_subjects=args.test_subjects,
                                            max_samples_per_subject=args.max_samples_per_subject)

    model = model.to(Config.DEVICE)
    model.eval()
    shown = 0
    with torch.no_grad():
        for batch in test_loader:
            if shown >= args.demo_samples:
                break

            frames = batch['frames'].to(Config.DEVICE)
            visemes = batch['visemes']
            phonemes = batch['phonemes']
            speaker_ids = model.speaker_ids_for(batch['subjects'], Config.DEVICE)

            predictions = model.predict(frames, speaker_ids=speaker_ids)
            fusion_weights = predictions['fusion_weights']
            if fusion_weights is not None:
                fusion_weights = torch.stack(fusion_weights, dim=0).mean(dim=0)  # avg over scales -> (B, T, 2)

            for i in range(frames.size(0)):
                if shown >= args.demo_samples:
                    break
                shown += 1

                valid = phonemes[i] != -1
                v_pred = predictions['viseme_predictions'][i].cpu()[valid].tolist()
                p_pred = predictions['phoneme_predictions'][i].cpu()[valid].tolist()
                v_true = visemes[i][valid].tolist()
                p_true = phonemes[i][valid].tolist()
                ambiguity = predictions['ambiguity_scores'][i].cpu()[valid].tolist()

                logger.info(f"\n{'=' * 50}")
                logger.info(f"DEMO SAMPLE {shown}: subject {batch['subjects'][i]}, utterance {batch['utterance_ids'][i]}")
                logger.info(f"Words: {' '.join(batch['words'][i])}")
                logger.info(f"{'=' * 50}")
                logger.info(f"Frames: {len(v_true)}")
                logger.info(f"Viseme Accuracy: {sum(p == t for p, t in zip(v_pred, v_true)) / len(v_true):.4f}")
                logger.info(f"Phoneme Accuracy: {sum(p == t for p, t in zip(p_pred, p_true)) / len(p_true):.4f}")
                logger.info(f"Average (predicted) Ambiguity: {sum(ambiguity) / len(ambiguity):.4f}")
                logger.info(f"Predicted phonemes: {collapse_phonemes(p_pred)}")
                logger.info(f"Target phonemes:    {collapse_phonemes(p_true)}")

                if fusion_weights is not None:
                    fw = fusion_weights[i].cpu()[valid]
                    logger.info(f"Fusion Weights - Visual: {fw[:, 0].mean():.4f}, Language: {fw[:, 1].mean():.4f}")

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

    logger.info(f"Synthetic data created for {list(subjects)} ({num_utterances} utterances each).")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enhanced Viseme-Phoneme Adaptive Model for Lip Reading")

    # Mode selection
    parser.add_argument('mode', choices=['train', 'evaluate', 'inference', 'analyze', 'create_sample'],
                        help='Operation mode')

    # Model configuration
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
    parser.add_argument('--num-workers', type=int, default=Config.NUM_WORKERS,
                        help='DataLoader worker processes (0 = load in the main process)')

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
    Config.NUM_WORKERS = args.num_workers
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
