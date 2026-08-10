import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from tqdm import tqdm
import logging
from src.config import Config

from models.fusion_network import VisemePhonemeAdaptiveModel
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

    def __init__(self, model, train_loader, val_loader, test_loader=None):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

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

    def train(self, num_epochs=Config.NUM_EPOCHS):
        """Full training loop"""
        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Training phase
            train_losses = self.train_epoch()
            logger.info(f"Epoch {epoch + 1} - Training Loss: {train_losses['total_loss']:.4f}")

            # Validation phase
            val_losses, viseme_acc, phoneme_acc = self.validate_epoch()

            # Learning rate scheduling
            self.scheduler.step(val_losses['total_loss'])

            # Save best model
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.save_checkpoint(f'best_model.pth', is_best=True)
                logger.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")

            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

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

        return {
            'test_loss': avg_test_losses['total_loss'],
            'viseme_accuracy': viseme_accuracy,
            'phoneme_accuracy': phoneme_accuracy,
            'high_ambiguity_accuracy': high_amb_accuracy,
            'low_ambiguity_accuracy': low_amb_accuracy
        }

    def save_checkpoint(self, filename, is_best=False):
        """Save model checkpoint"""
        os.makedirs('checkpoints', exist_ok=True)
        filepath = os.path.join('checkpoints', filename)

        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': {k: v for k, v in Config.__dict__.items() if not callable(v) and not k.startswith('__')}

        }

        torch.save(checkpoint, filepath)

        if is_best:
            best_path = os.path.join('checkpoints', 'best_model.pth')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=Config.DEVICE)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']

        logger.info(f"Checkpoint loaded from epoch {self.epoch}")


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