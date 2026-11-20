import torch
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

from models.fusion_network import VisemePhonemeAdaptiveModel, ModelFactory
from models.visual_encoder import upgrade_legacy_state_dict
from src.data_loader import create_data_loaders
from src.viseme_mapping import viseme_mapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SILENCE_VISEME = viseme_mapper.get_viseme('sil')


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


class AccuracyMeter:
    """Frame-level viseme/phoneme accuracy over valid (non-padding) frames.

    Also reports accuracy on speech frames only (target viseme != silence): silence is a large
    share of every GRID clip and easy to predict, so all-frame accuracy alone overstates how well
    speech is being read. Optional ground-truth ambiguity scores give a high/low ambiguity split
    (analysis only — they are never fed to the model).
    """

    def __init__(self):
        self.counts = {k: 0 for k in ('total', 'viseme', 'phoneme', 'speech_total', 'speech_viseme',
                                      'speech_phoneme', 'high_total', 'high_phoneme', 'low_total',
                                      'low_phoneme')}

    def update(self, viseme_logits, phoneme_logits, visemes, phonemes, ambiguity_scores=None):
        viseme_ok = torch.argmax(viseme_logits, dim=-1) == visemes
        phoneme_ok = torch.argmax(phoneme_logits, dim=-1) == phonemes
        valid = (visemes != -1) & (phonemes != -1)
        speech = valid & (visemes != SILENCE_VISEME)

        c = self.counts
        c['total'] += valid.sum().item()
        c['viseme'] += viseme_ok[valid].sum().item()
        c['phoneme'] += phoneme_ok[valid].sum().item()
        c['speech_total'] += speech.sum().item()
        c['speech_viseme'] += viseme_ok[speech].sum().item()
        c['speech_phoneme'] += phoneme_ok[speech].sum().item()
        if ambiguity_scores is not None:
            high = valid & (ambiguity_scores > Config.AMBIGUITY_THRESHOLD)
            low = valid & (ambiguity_scores <= Config.AMBIGUITY_THRESHOLD)
            c['high_total'] += high.sum().item()
            c['high_phoneme'] += phoneme_ok[high].sum().item()
            c['low_total'] += low.sum().item()
            c['low_phoneme'] += phoneme_ok[low].sum().item()

    @staticmethod
    def _ratio(num, den):
        return num / den if den > 0 else 0.0

    def results(self):
        c = self.counts
        return {
            'viseme_accuracy': self._ratio(c['viseme'], c['total']),
            'phoneme_accuracy': self._ratio(c['phoneme'], c['total']),
            'speech_viseme_accuracy': self._ratio(c['speech_viseme'], c['speech_total']),
            'speech_phoneme_accuracy': self._ratio(c['speech_phoneme'], c['speech_total']),
            'high_ambiguity_accuracy': self._ratio(c['high_phoneme'], c['high_total']),
            'low_ambiguity_accuracy': self._ratio(c['low_phoneme'], c['low_total']),
            'total_frames': c['total'],
            'speech_frames': c['speech_total'],
        }


class LipReadingTrainer:
    """Training class for the viseme-phoneme adaptive model"""

    LOSS_KEYS = ('total_loss', 'viseme_loss', 'aux_viseme_loss', 'phoneme_loss', 'word_loss', 'fusion_reg')

    METRIC_FIELDS = [
        'epoch', 'lr', 'epoch_seconds',
        'train_total_loss', 'train_viseme_loss', 'train_aux_viseme_loss', 'train_phoneme_loss',
        'train_word_loss', 'train_fusion_reg',
        'val_total_loss', 'val_viseme_loss', 'val_aux_viseme_loss', 'val_phoneme_loss', 'val_word_loss',
        'val_viseme_accuracy', 'val_phoneme_accuracy',
        'val_speech_viseme_accuracy', 'val_speech_phoneme_accuracy',
    ]

    def __init__(self, model, train_loader, val_loader, test_loader=None, checkpoint_dir=None):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Speakers seen in training; other speakers use the "unknown speaker" embedding
        self.model.known_speakers = list(getattr(train_loader.dataset, 'subjects', []))

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

        logger.info(f"Trainer initialized. Model type: {self.model.model_type}, "
                    f"{self._count_parameters():,} trainable parameters, known speakers: {self.model.known_speakers}")
        logger.info(f"Training on {len(train_loader)} batches, validating on {len(val_loader)} batches.")
        logger.info(f"Checkpoints and metrics will be written to: {os.path.abspath(self.checkpoint_dir)}")

    def _get_timestamp(self):
        """Get current timestamp for logging"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _count_parameters(self):
        """Count trainable parameters in model"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _run_batch(self, batch):
        """Forward pass + losses for one batch. Only frames and speakers go into the model."""
        frames = batch['frames'].to(Config.DEVICE)
        targets = {
            'visemes': batch['visemes'].to(Config.DEVICE),
            'phonemes': batch['phonemes'].to(Config.DEVICE),
            'words': batch['word_labels'].to(Config.DEVICE),
        }
        speaker_ids = self.model.speaker_ids_for(batch['subjects'], Config.DEVICE)

        predictions = self.model(frames=frames, speaker_ids=speaker_ids)
        losses = self.model.compute_loss(predictions, targets)
        return predictions, targets, losses

    def _evaluate_loader(self, loader, desc):
        """Average losses and accuracy metrics over a loader (no grad)."""
        self.model.eval()
        epoch_losses = {k: [] for k in self.LOSS_KEYS}
        meter = AccuracyMeter()

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                predictions, targets, losses = self._run_batch(batch)
                for key, value in losses.items():
                    if key in epoch_losses:
                        epoch_losses[key].append(value.item())
                meter.update(predictions['viseme_logits'], predictions['phoneme_logits'],
                             targets['visemes'], targets['phonemes'],
                             batch['ambiguity_scores'].to(Config.DEVICE))

        avg_losses = {key: float(np.mean(values)) for key, values in epoch_losses.items() if values}
        return avg_losses, meter.results()

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {k: [] for k in self.LOSS_KEYS}

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            predictions, _, losses = self._run_batch(batch)

            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for key, value in losses.items():
                if key in epoch_losses:
                    epoch_losses[key].append(value.item())

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

                if predictions['fusion_weights'] is not None:
                    fusion_weights = torch.stack(predictions['fusion_weights'], dim=-2).detach().cpu()
                    self.writer.add_scalar('Train/avg_visual_weight', fusion_weights[..., 0].mean(), global_step)
                    self.writer.add_scalar('Train/avg_language_weight', fusion_weights[..., 1].mean(), global_step)

        avg_losses = {key: float(np.mean(values)) for key, values in epoch_losses.items() if values}
        self.train_losses.append(avg_losses['total_loss'])
        return avg_losses

    def validate_epoch(self):
        """Validate for one epoch"""
        avg_losses, metrics = self._evaluate_loader(self.val_loader, "Validation")
        self.val_losses.append(avg_losses['total_loss'])

        for key, value in avg_losses.items():
            self.writer.add_scalar(f'Val/{key}', value, self.epoch)
        for key in ('viseme_accuracy', 'phoneme_accuracy', 'speech_viseme_accuracy', 'speech_phoneme_accuracy'):
            self.writer.add_scalar(f'Val/{key}', metrics[key], self.epoch)

        logger.info(f"Validation - Loss: {avg_losses['total_loss']:.4f}, "
                    f"Viseme Acc: {metrics['viseme_accuracy']:.4f}, Phoneme Acc: {metrics['phoneme_accuracy']:.4f} | "
                    f"speech-only Viseme Acc: {metrics['speech_viseme_accuracy']:.4f}, "
                    f"Phoneme Acc: {metrics['speech_phoneme_accuracy']:.4f}")

        return avg_losses, metrics

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

            train_losses = self.train_epoch()
            logger.info(f"Epoch {epoch + 1} - Training Loss: {train_losses['total_loss']:.4f}, "
                        f"Viseme Loss: {train_losses.get('viseme_loss', float('nan')):.4f}, "
                        f"Phoneme Loss: {train_losses.get('phoneme_loss', float('nan')):.4f}")

            val_losses, val_metrics = self.validate_epoch()

            self.scheduler.step(val_losses['total_loss'])

            row = {'epoch': epoch + 1, 'lr': self.optimizer.param_groups[0]['lr'],
                   'epoch_seconds': round(time.time() - epoch_start, 1)}
            row.update({f'train_{k}': v for k, v in train_losses.items()})
            row.update({f'val_{k}': v for k, v in val_losses.items()})
            row.update({f'val_{k}': val_metrics[k] for k in ('viseme_accuracy', 'phoneme_accuracy',
                                                             'speech_viseme_accuracy', 'speech_phoneme_accuracy')})
            self._write_metrics_row(row)

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

            self.early_stopping(val_losses['total_loss'])
            if self.early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        logger.info("Training completed!")

        if self.test_loader is not None:
            # Test the best (lowest validation loss) model, not whatever the last epoch produced
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            if os.path.exists(best_path):
                best = torch.load(best_path, map_location=Config.DEVICE, weights_only=False)
                self.model.load_state_dict(best['model_state_dict'])
                self.epoch = best['epoch']
                logger.info(f"Loaded best model (epoch {self.epoch + 1}) for test evaluation")
            self.evaluate()

        self.writer.close()

    def evaluate(self):
        """Evaluate model on test set"""
        logger.info("Evaluating on test set...")

        avg_losses, metrics = self._evaluate_loader(self.test_loader, "Testing")

        logger.info("=" * 50)
        logger.info("TEST RESULTS")
        logger.info("=" * 50)
        logger.info(f"Test Loss: {avg_losses['total_loss']:.4f}")
        logger.info(f"Viseme Accuracy: {metrics['viseme_accuracy']:.4f}")
        logger.info(f"Phoneme Accuracy: {metrics['phoneme_accuracy']:.4f}")
        logger.info(f"Speech-only Viseme Accuracy: {metrics['speech_viseme_accuracy']:.4f}")
        logger.info(f"Speech-only Phoneme Accuracy: {metrics['speech_phoneme_accuracy']:.4f}")
        logger.info(f"High Ambiguity Phoneme Accuracy: {metrics['high_ambiguity_accuracy']:.4f}")
        logger.info(f"Low Ambiguity Phoneme Accuracy: {metrics['low_ambiguity_accuracy']:.4f}")
        logger.info("=" * 50)

        test_results = {'test_loss': avg_losses['total_loss'], **metrics,
                        'epoch': self.epoch + 1, 'model_type': self.model.model_type}
        with open(self.test_results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
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
            'model_type': self.model.model_type,
            'known_speakers': self.model.known_speakers,
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
        # Python scalars/lists, which the PyTorch>=2.6 weights_only default rejects.
        checkpoint = torch.load(filepath, map_location=Config.DEVICE, weights_only=False)

        saved_type = checkpoint.get('model_type')
        if saved_type and saved_type != self.model.model_type:
            raise ValueError(f"Checkpoint was trained as '{saved_type}' but the model is "
                             f"'{self.model.model_type}'; pass --model-type {saved_type} to resume.")

        self.model.load_state_dict(upgrade_legacy_state_dict(checkpoint['model_state_dict']))
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
    train_loader, val_loader, test_loader = create_data_loaders(
        train_subjects=['s1'],
        val_subjects=['s2'],
        test_subjects=['s3']
    )

    model = ModelFactory.create_full_model()
    trainer = LipReadingTrainer(model, train_loader, val_loader, test_loader)

    logger.info("Viseme mappings:")
    viseme_mapper.print_mappings()

    trainer.train()


if __name__ == "__main__":
    main()
