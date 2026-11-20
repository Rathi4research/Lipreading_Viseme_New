import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import csv
import json
import os
import time
import numpy as np
from tqdm import tqdm
import logging
from src.config import Config

from models.fusion_network import VisemePhonemeAdaptiveModel
from models.visual_encoder import materialize_lazy_layers
from src.data_loader import create_data_loaders
from src.viseme_mapping import viseme_mapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class LipReadingTrainer:
    """Training class for the viseme-phoneme adaptive model"""

    METRIC_FIELDS = [
        'epoch', 'lr', 'epoch_seconds',
        'train_total_loss', 'train_viseme_loss', 'train_phoneme_loss', 'train_fusion_reg',
        'val_total_loss', 'val_viseme_loss', 'val_phoneme_loss',
        'val_viseme_accuracy', 'val_phoneme_accuracy',
    ]

    def __init__(self, model, train_loader, val_loader, test_loader=None, checkpoint_dir=None):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Where checkpoints, the per-epoch metrics CSV and the test results JSON are written.
        # Defaults to Config.CHECKPOINT_DIR (Google Drive on Colab); main.py --local-checkpoints
        # / --checkpoint-dir override it. Created up front so a bad path fails before training.
        self.checkpoint_dir = checkpoint_dir or Config.CHECKPOINT_DIR
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.checkpoint_dir, 'training_metrics.csv')
        self.test_results_path = os.path.join(self.checkpoint_dir, 'test_results.json')
        self.resumed = False

        # Optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=1e-5
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,

        )

        # Early stopping
        self.early_stopping = EarlyStopping(patience=Config.PATIENCE)

        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=f'runs/lipreading_{self._get_timestamp()}')

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epoch = 0

        logger.info(f"Trainer initialized. Model has {self._count_parameters()} parameters.")
        logger.info(f"Training on {len(train_loader)} batches, validating on {len(val_loader)} batches.")
        logger.info(f"Checkpoints and metrics will be written to: {os.path.abspath(self.checkpoint_dir)}")

    def _get_timestamp(self):
        """Get current timestamp for logging"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _count_parameters(self):
        """Count trainable parameters in model"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {
            'total_loss': [],
            'viseme_loss': [],
            'phoneme_loss': [],
            'word_loss': [],
            'fusion_reg': []
        }

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            frames = batch['frames'].to(Config.DEVICE)
            visemes = batch['visemes'].to(Config.DEVICE)
            phonemes = batch['phonemes'].to(Config.DEVICE)
            ambiguity_scores = batch['ambiguity_scores'].to(Config.DEVICE)

            # Create speaker IDs (simple mapping s1=0, s2=1, s3=2)
            speaker_ids = torch.tensor([
                Config.SPEAKER_TO_ID.get(s, 0)
                for s in batch['subjects']
            ]).to(Config.DEVICE)

            # Forward pass
            predictions = self.model(
                frames=frames,
                phoneme_targets=phonemes,
                speaker_ids=speaker_ids,
                ambiguity_scores=ambiguity_scores
            )

            # Calculate loss
            targets = {
                'visemes': visemes,
                'phonemes': phonemes
            }

            losses = self.model.compute_loss(predictions, targets)

            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Store losses
            for key, value in losses.items():
                if key in epoch_losses:
                    epoch_losses[key].append(value.item())

            # Update progress bar
            progress_bar.set_postfix({
                'total_loss': f"{losses['total_loss'].item():.4f}",
                'viseme_loss': f"{losses.get('viseme_loss', 0):.4f}",
                'phoneme_loss': f"{losses.get('phoneme_loss', 0):.4f}"
            })

            # Log to tensorboard every 100 batches
            if batch_idx % 100 == 0:
                global_step = self.epoch * len(self.train_loader) + batch_idx
                for key, value in losses.items():
                    self.writer.add_scalar(f'Train/{key}', value.item(), global_step)

                # Log fusion weights distribution
                if predictions['fusion_weights'] is not None:
                    # Stack the list of tensors into a single tensor
                    if isinstance(predictions['fusion_weights'], list):
                        fusion_weights = torch.stack(predictions['fusion_weights'],
                                                     dim=-2)  # shape: [B, T, num_scales, 2]
                    else:
                        fusion_weights = predictions['fusion_weights']
                    fusion_weights = fusion_weights.detach().cpu()

                    # Log average weights per modality
                    visual_weights = fusion_weights[..., 0].mean()
                    language_weights = fusion_weights[..., 1].mean()
                    self.writer.add_scalar('Train/avg_visual_weight', visual_weights, global_step)
                    self.writer.add_scalar('Train/avg_language_weight', language_weights, global_step)

        # Calculate average losses for the epoch
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items() if values}
        self.train_losses.append(avg_losses['total_loss'])

        return avg_losses

    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        epoch_losses = {
            'total_loss': [],
            'viseme_loss': [],
            'phoneme_loss': [],
            'word_loss': []
        }

        # Metrics tracking
        correct_visemes = 0
        correct_phonemes = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                frames = batch['frames'].to(Config.DEVICE)
                visemes = batch['visemes'].to(Config.DEVICE)
                phonemes = batch['phonemes'].to(Config.DEVICE)
                ambiguity_scores = batch['ambiguity_scores'].to(Config.DEVICE)

                speaker_ids = torch.tensor([
                    Config.SPEAKER_TO_ID.get(s, 0)
                    for s in batch['subjects']
                ]).to(Config.DEVICE)

                # Forward pass
                predictions = self.model(
                    frames=frames,
                    phoneme_targets=phonemes,
                    speaker_ids=speaker_ids,
                    ambiguity_scores=ambiguity_scores
                )

                # Calculate loss
                targets = {
                    'visemes': visemes,
                    'phonemes': phonemes
                }

                losses = self.model.compute_loss(predictions, targets)

                # Store losses
                for key, value in losses.items():
                    if key in epoch_losses:
                        epoch_losses[key].append(value.item())

                # Calculate accuracy
                viseme_preds = torch.argmax(predictions['viseme_logits'], dim=-1)
                phoneme_preds = torch.argmax(predictions['phoneme_logits'], dim=-1)

                # Mask out padding tokens (-1)
                valid_mask = (visemes != -1) & (phonemes != -1)

                correct_visemes += (viseme_preds == visemes)[valid_mask].sum().item()
                correct_phonemes += (phoneme_preds == phonemes)[valid_mask].sum().item()
                total_predictions += valid_mask.sum().item()

        # Calculate average losses and accuracies
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items() if values}

        viseme_accuracy = correct_visemes / total_predictions if total_predictions > 0 else 0
        phoneme_accuracy = correct_phonemes / total_predictions if total_predictions > 0 else 0

        self.val_losses.append(avg_losses['total_loss'])

        # Log to tensorboard
        for key, value in avg_losses.items():
            self.writer.add_scalar(f'Val/{key}', value, self.epoch)

        self.writer.add_scalar('Val/viseme_accuracy', viseme_accuracy, self.epoch)
        self.writer.add_scalar('Val/phoneme_accuracy', phoneme_accuracy, self.epoch)

        logger.info(f"Validation - Loss: {avg_losses['total_loss']:.4f}, "
                    f"Viseme Acc: {viseme_accuracy:.4f}, Phoneme Acc: {phoneme_accuracy:.4f}")

        return avg_losses, viseme_accuracy, phoneme_accuracy

    def _write_metrics_row(self, row):
        """Append one epoch's metrics to training_metrics.csv (header written on first row)."""
        write_header = not os.path.exists(self.metrics_path)
        with open(self.metrics_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.METRIC_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, '') for k in self.METRIC_FIELDS})

    def train(self, num_epochs=Config.NUM_EPOCHS):
        """Full training loop"""
        # Fresh run: start a new metrics file. Resumed run: keep appending to the existing one.
        start_epoch = self.epoch + 1 if self.resumed else 0
        if not self.resumed and os.path.exists(self.metrics_path):
            os.remove(self.metrics_path)

        logger.info(f"Starting training for epochs {start_epoch + 1}..{num_epochs}")

        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch
            epoch_start = time.time()

            # Training phase
            train_losses = self.train_epoch()
            logger.info(f"Epoch {epoch + 1} - Training Loss: {train_losses['total_loss']:.4f}, "
                        f"Viseme Loss: {train_losses.get('viseme_loss', float('nan')):.4f}, "
                        f"Phoneme Loss: {train_losses.get('phoneme_loss', float('nan')):.4f}")

            # Validation phase
            val_losses, viseme_acc, phoneme_acc = self.validate_epoch()

            # Learning rate scheduling
            self.scheduler.step(val_losses['total_loss'])

            self._write_metrics_row({
                'epoch': epoch + 1,
                'lr': self.optimizer.param_groups[0]['lr'],
                'epoch_seconds': round(time.time() - epoch_start, 1),
                'train_total_loss': train_losses.get('total_loss'),
                'train_viseme_loss': train_losses.get('viseme_loss'),
                'train_phoneme_loss': train_losses.get('phoneme_loss'),
                'train_fusion_reg': train_losses.get('fusion_reg'),
                'val_total_loss': val_losses.get('total_loss'),
                'val_viseme_loss': val_losses.get('viseme_loss'),
                'val_phoneme_loss': val_losses.get('phoneme_loss'),
                'val_viseme_accuracy': viseme_acc,
                'val_phoneme_accuracy': phoneme_acc,
            })

            # Save best model
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.save_checkpoint('best_model.pth')
                logger.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")

            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

            # Always keep the latest state so an interrupted (e.g. Colab) session can resume
            self.save_checkpoint('last_model.pth')

            # Early stopping
            self.early_stopping(val_losses['total_loss'])
            if self.early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        logger.info("Training completed!")

        # Final evaluation
        if self.test_loader is not None:
            self.evaluate()

        self.writer.close()

    def evaluate(self):
        """Evaluate model on test set"""
        logger.info("Evaluating on test set...")

        self.model.eval()
        test_losses = {
            'total_loss': [],
            'viseme_loss': [],
            'phoneme_loss': []
        }

        correct_visemes = 0
        correct_phonemes = 0
        total_predictions = 0

        # Per-ambiguity analysis
        high_ambiguity_correct = 0
        high_ambiguity_total = 0
        low_ambiguity_correct = 0
        low_ambiguity_total = 0

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                frames = batch['frames'].to(Config.DEVICE)
                visemes = batch['visemes'].to(Config.DEVICE)
                phonemes = batch['phonemes'].to(Config.DEVICE)
                ambiguity_scores = batch['ambiguity_scores'].to(Config.DEVICE)

                speaker_ids = torch.tensor([
                    Config.SPEAKER_TO_ID.get(s, 0)
                    for s in batch['subjects']
                ]).to(Config.DEVICE)

                predictions = self.model(
                    frames=frames,
                    phoneme_targets=phonemes,
                    speaker_ids=speaker_ids,
                    ambiguity_scores=ambiguity_scores
                )

                targets = {
                    'visemes': visemes,
                    'phonemes': phonemes
                }

                losses = self.model.compute_loss(predictions, targets)

                for key, value in losses.items():
                    if key in test_losses:
                        test_losses[key].append(value.item())

                # Calculate accuracy
                viseme_preds = torch.argmax(predictions['viseme_logits'], dim=-1)
                phoneme_preds = torch.argmax(predictions['phoneme_logits'], dim=-1)

                valid_mask = (visemes != -1) & (phonemes != -1)

                correct_visemes += (viseme_preds == visemes)[valid_mask].sum().item()
                correct_phonemes += (phoneme_preds == phonemes)[valid_mask].sum().item()
                total_predictions += valid_mask.sum().item()

                # Ambiguity-based analysis
                high_ambiguity_mask = (ambiguity_scores > Config.AMBIGUITY_THRESHOLD) & valid_mask
                low_ambiguity_mask = (ambiguity_scores <= Config.AMBIGUITY_THRESHOLD) & valid_mask

                high_ambiguity_correct += (phoneme_preds == phonemes)[high_ambiguity_mask].sum().item()
                high_ambiguity_total += high_ambiguity_mask.sum().item()

                low_ambiguity_correct += (phoneme_preds == phonemes)[low_ambiguity_mask].sum().item()
                low_ambiguity_total += low_ambiguity_mask.sum().item()

        # Calculate metrics
        avg_test_losses = {key: np.mean(values) for key, values in test_losses.items() if values}

        viseme_accuracy = correct_visemes / total_predictions
        phoneme_accuracy = correct_phonemes / total_predictions

        high_amb_accuracy = high_ambiguity_correct / high_ambiguity_total if high_ambiguity_total > 0 else 0
        low_amb_accuracy = low_ambiguity_correct / low_ambiguity_total if low_ambiguity_total > 0 else 0

        # Log results
        logger.info("=" * 50)
        logger.info("TEST RESULTS")
        logger.info("=" * 50)
        logger.info(f"Test Loss: {avg_test_losses['total_loss']:.4f}")
        logger.info(f"Viseme Accuracy: {viseme_accuracy:.4f}")
        logger.info(f"Phoneme Accuracy: {phoneme_accuracy:.4f}")
        logger.info(f"High Ambiguity Phoneme Accuracy: {high_amb_accuracy:.4f}")
        logger.info(f"Low Ambiguity Phoneme Accuracy: {low_amb_accuracy:.4f}")
        logger.info("=" * 50)

        test_results = {
            'test_loss': avg_test_losses['total_loss'],
            'viseme_accuracy': viseme_accuracy,
            'phoneme_accuracy': phoneme_accuracy,
            'high_ambiguity_accuracy': high_amb_accuracy,
            'low_ambiguity_accuracy': low_amb_accuracy,
            'total_frames': total_predictions,
            'epoch': self.epoch + 1,
        }

        with open(self.test_results_path, 'w') as f:
            json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in test_results.items()}, f, indent=2)
        logger.info(f"Test results written to {self.test_results_path}")

        return test_results

    @staticmethod
    def _config_snapshot():
        """Plain-data copy of Config (picklable; skips methods/classmethods)."""
        simple = (str, int, float, bool, list, tuple, dict, type(None))
        return {k: v for k, v in vars(Config).items()
                if not k.startswith('__') and isinstance(v, simple)}

    def save_checkpoint(self, filename):
        """Save model checkpoint to self.checkpoint_dir"""
        filepath = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self._config_snapshot()
        }

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """Load model checkpoint (to resume training)"""
        # weights_only=False: our own checkpoints also hold optimizer/scheduler state and
        # numpy scalars, which the PyTorch>=2.6 weights_only default rejects.
        checkpoint = torch.load(filepath, map_location=Config.DEVICE, weights_only=False)

        # The visual encoder builds fc1_dynamic/fc2_dynamic lazily on the first forward pass,
        # so a freshly constructed model doesn't have them yet — create them before loading.
        materialize_lazy_layers(self.model, checkpoint['model_state_dict'])

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.resumed = True

        logger.info(f"Checkpoint loaded from epoch {self.epoch + 1}")


def main():
    """Main training function"""
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_subjects=['s1'],
        val_subjects=['s2'],
        test_subjects=['s3']
    )

    # Create model
    model = VisemePhonemeAdaptiveModel()

    # Create trainer
    trainer = LipReadingTrainer(model, train_loader, val_loader, test_loader)

    # Print viseme mappings for reference
    logger.info("Viseme mappings:")
    viseme_mapper.print_mappings()

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()