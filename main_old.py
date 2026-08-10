# Import project modules
from src.config import Config
import logging
import os
import torch

import sys
import argparse
from src.data_loader import create_data_loaders
from src.trainer import LipReadingTrainer
from src.evaluator import LipReadingEvaluator
from src.viseme_mapping import viseme_mapper
from src.adaptive_fusion import MultiScaleAdaptiveFusion, ConfusionMatrixDecoder

from src.language_model import ContextualLanguageModel


from models.fusion_network import VisemePhonemeAdaptiveModel, ModelFactory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lipreading.log'),
        logging.StreamHandler(sys.stdout)
    ]
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


def train_model(args):
    """Train the model"""
    logger.info("Starting model training...")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_subjects=args.train_subjects,
        val_subjects=args.val_subjects,
        test_subjects=args.test_subjects
    )

    # Create model
    if args.model_type == 'baseline':
        model = ModelFactory.create_baseline_model()
    elif args.model_type == 'visual_only':
        model = ModelFactory.create_visual_only_model()
    else:
        model = ModelFactory.create_full_model()

    print_model_info(model)

    # Print viseme mappings
    logger.info("Viseme-Phoneme Mappings:")
    viseme_mapper.print_mappings()

    # Create trainer
    trainer = LipReadingTrainer(model, train_loader, val_loader, test_loader)

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
    _, _, test_loader = create_data_loaders(test_subjects=args.test_subjects)

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
        # Plot viseme confusion matrix
        viseme_labels = [f"V{i}" for i in range(viseme_mapper.num_visemes)]
        evaluator.plot_confusion_matrix(
            results['confusion_matrices']['viseme_confusion_matrix'],
            viseme_labels,
            "Viseme Confusion Matrix",
            "plots/viseme_confusion_matrix.png"
        )

        # Plot phoneme confusion matrix (sample of phonemes)
        phoneme_labels = [f"P{i}" for i in range(min(20, len(Config.PHONEMES)))]  # First 20 phonemes
        cm_subset = results['confusion_matrices']['phoneme_confusion_matrix'][:20, :20]
        evaluator.plot_confusion_matrix(
            cm_subset,
            phoneme_labels,
            "Phoneme Confusion Matrix (First 20 Phonemes)",
            "plots/phoneme_confusion_matrix.png"
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
    """Run inference demo"""
    logger.info("Running inference demo...")

    # Load model
    if args.checkpoint_path:
        model = ModelFactory.load_pretrained_model(args.checkpoint_path, args.model_type)
        logger.info(f"Loaded model from {args.checkpoint_path}")
    else:
        logger.error("No checkpoint path provided for inference")
        return

    # Create a small demo dataset
    _, _, test_loader = create_data_loaders(test_subjects=['s3'])

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= args.demo_samples:
                break

            frames = batch['frames'].to(Config.DEVICE)
            visemes = batch['visemes'].to(Config.DEVICE)
            phonemes = batch['phonemes'].to(Config.DEVICE)
            subjects = batch['subjects']
            utterance_ids = batch['utterance_ids']

            # Run inference
            predictions = model.predict(frames)

            # Process results for each sample in batch
            batch_size = frames.size(0)
            for i in range(min(batch_size, args.demo_samples - batch_idx)):
                logger.info(f"\n{'=' * 50}")
                logger.info(f"DEMO SAMPLE {batch_idx * Config.BATCH_SIZE + i + 1}")
                logger.info(f"Subject: {subjects[i]}, Utterance: {utterance_ids[i]}")
                logger.info(f"{'=' * 50}")

                # Get predictions for this sample
                viseme_pred = predictions['viseme_predictions'][i]
                phoneme_pred = predictions['phoneme_predictions'][i]
                viseme_probs = predictions['viseme_probabilities'][i]
                phoneme_probs = predictions['phoneme_probabilities'][i]
                ambiguity = predictions['ambiguity_scores'][i]

                # Remove padding and convert to lists
                valid_length = (phonemes[i] != -1).sum().item()

                viseme_pred_list = viseme_pred[:valid_length].cpu().tolist()
                phoneme_pred_list = phoneme_pred[:valid_length].cpu().tolist()
                viseme_target_list = visemes[i, :valid_length].cpu().tolist()
                phoneme_target_list = phonemes[i, :valid_length].cpu().tolist()
                ambiguity_list = ambiguity[:valid_length].cpu().tolist()

                # Calculate accuracy
                viseme_acc = sum(1 for p, t in zip(viseme_pred_list, viseme_target_list) if p == t) / len(
                    viseme_pred_list)
                phoneme_acc = sum(1 for p, t in zip(phoneme_pred_list, phoneme_target_list) if p == t) / len(
                    phoneme_pred_list)

                logger.info(f"Sequence Length: {valid_length}")
                logger.info(f"Viseme Accuracy: {viseme_acc:.4f}")
                logger.info(f"Phoneme Accuracy: {phoneme_acc:.4f}")
                logger.info(f"Average Ambiguity: {sum(ambiguity_list) / len(ambiguity_list):.4f}")

                # Show first 10 predictions vs targets
                logger.info("\nFirst 10 Predictions vs Targets:")
                logger.info("Pos | V_Pred V_True | P_Pred P_True | Ambiguity | Correct?")
                logger.info("-" * 65)

                for j in range(min(10, len(viseme_pred_list))):
                    v_pred = viseme_pred_list[j]
                    v_true = viseme_target_list[j]
                    p_pred = phoneme_pred_list[j]
                    p_true = phoneme_target_list[j]
                    amb = ambiguity_list[j]

                    v_correct = "✓" if v_pred == v_true else "✗"
                    p_correct = "✓" if p_pred == p_true else "✗"

                    logger.info(
                        f"{j:2d}  |   {v_pred:2d}    {v_true:2d}   |   {p_pred:2d}    {p_true:2d}   |   {amb:.3f}   | {v_correct} {p_correct}")

                # Show fusion weights if available
                if predictions['fusion_weights'] is not None:
                    fusion_weights = predictions['fusion_weights'][i, :valid_length]
                    visual_weights = fusion_weights[:, 0].cpu().tolist()
                    language_weights = fusion_weights[:, 1].cpu().tolist()

                    avg_visual = sum(visual_weights) / len(visual_weights)
                    avg_language = sum(language_weights) / len(language_weights)

                    logger.info(f"\nFusion Weights - Visual: {avg_visual:.4f}, Language: {avg_language:.4f}")

    logger.info("Inference demo completed!")


def analyze_dataset(args):
    """Analyze the dataset"""
    logger.info("Analyzing dataset...")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_subjects=args.train_subjects,
        val_subjects=args.val_subjects,
        test_subjects=args.test_subjects
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

    # Ambiguity distribution
    valid_ambiguity = ambiguity_scores[ambiguity_scores >= 0]  # Remove padding
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


def create_sample_data():
    """Create sample data files for testing"""
    logger.info("Creating sample data files...")

    # Create directories
    for subject in ['s1', 's2', 's3']:
        os.makedirs(f'data/videos/{subject}', exist_ok=True)
        os.makedirs(f'data/alignments/{subject}', exist_ok=True)

    # Create sample alignment files
    sample_alignments = {
        'bbaf2n': [
            '0.0 0.2 sil',
            '0.2 0.4 b',
            '0.4 0.6 i',
            '0.6 0.8 n',
            '0.8 1.0 sil',
            '1.0 1.2 b',
            '1.2 1.4 l',
            '1.4 1.6 u',
            '1.6 1.8 sil'
        ],
        'bbaf3s': [
            '0.0 0.15 sil',
            '0.15 0.3 b',
            '0.3 0.45 i',
            '0.45 0.6 n',
            '0.6 0.75 sil',
            '0.75 0.9 r',
            '0.9 1.05 e',
            '1.05 1.2 d',
            '1.2 1.35 sil'
        ]
    }

    for subject in ['s1', 's2', 's3']:
        for utterance_id, alignment in sample_alignments.items():
            align_path = f'data/alignments/{subject}/{utterance_id}.align'
            with open(align_path, 'w') as f:
                f.write('\n'.join(alignment))

            # Create dummy video file (just create empty file as placeholder)
            video_path = f'data/videos/{subject}/{utterance_id}.mp4'
            with open(video_path, 'w') as f:
                f.write('')  # Placeholder - would need actual video data

    logger.info("Sample data files created!")
    logger.info("Note: Video files are placeholders. Replace with actual .mp4 files.")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enhanced Viseme-Phoneme Adaptive Model for Lip Reading")

    # Mode selection
    parser.add_argument('mode', choices=['train', 'evaluate', 'inference', 'analyze', 'create_sample'],
                        help='Operation mode')

    # Model configuration
    parser.add_argument('--model-type', choices=['full', 'baseline', 'visual_only'],
                        default='full', help='Model type to use')
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

    # Inference configuration
    parser.add_argument('--demo-samples', type=int, default=5,
                        help='Number of samples for demo')

    # Device configuration
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')

    args = parser.parse_args()

    # Setup
    setup_directories()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    Config.DEVICE = device

    # Update config with command line arguments
    if hasattr(args, 'batch_size'):
        Config.BATCH_SIZE = args.batch_size
    if hasattr(args, 'learning_rate'):
        Config.LEARNING_RATE = args.learning_rate

    logger.info(f"Using device: {Config.DEVICE}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model type: {args.model_type}")

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