import csv
import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import Config
from src.data_loader import create_data_loaders
from src.metrics import autocast_context, batch_to_device, run_evaluation
from src.viseme_mapping import viseme_mapper
from models.fusion_network import ModelFactory
from models.visual_encoder import upgrade_legacy_state_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cosine_with_warmup(total_steps, warmup_steps, min_ratio):
    """LR multiplier: linear warmup to 1, then cosine decay to min_ratio."""
    def schedule(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return schedule


class LipReadingTrainer:
    """Training loop for the viseme-phoneme adaptive model.

    Model selection and early stopping use the validation greedy word error rate (ties broken by
    the validation CTC loss). The final test evaluates the best checkpoint with greedy and
    grammar-constrained decoding, homophene metrics and the fusion-gate analysis.
    """

    LOSS_KEYS = ('total_loss', 'ctc_loss', 'viseme_loss', 'aux_viseme_loss', 'phoneme_loss', 'word_loss', 'fusion_reg')
    METRIC_FIELDS = (['epoch', 'lr', 'epoch_seconds'] + [f'train_{k}' for k in LOSS_KEYS] +
                     [f'val_{k}' for k in LOSS_KEYS] +
                     ['val_wer', 'val_cer', 'val_sentence_accuracy', 'val_viseme_accuracy', 'val_phoneme_accuracy',
                      'val_speech_viseme_accuracy', 'val_speech_phoneme_accuracy', 'best'])

    def __init__(self, model, train_loader, val_loader, test_loader=None, checkpoint_dir=None,
                 num_epochs=Config.NUM_EPOCHS, run_info=None):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.run_info = run_info or {}

        # Speakers seen in training; other speakers use the "unknown speaker" embedding
        self.model.known_speakers = list(getattr(train_loader.dataset, 'subjects', []))

        self.checkpoint_dir = checkpoint_dir or Config.CHECKPOINT_DIR
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.checkpoint_dir, 'training_metrics.csv')
        self.test_results_path = os.path.join(self.checkpoint_dir, 'test_results.json')
        self.resumed = False

        self.optimizer = optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        total_steps = max(1, num_epochs * len(train_loader))
        warmup = max(1, int(round(Config.WARMUP_RATIO * total_steps)))
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, cosine_with_warmup(total_steps, warmup, Config.MIN_LR_RATIO))

        self.use_amp = Config.AMP and str(Config.DEVICE).startswith('cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.writer = SummaryWriter(log_dir=f'runs/lipreading_{time.strftime("%Y%m%d_%H%M%S")}')

        self.train_losses, self.val_losses = [], []
        self.best_key = (float('inf'), float('inf'))   # (val WER, val CTC loss)
        self.epochs_without_improvement = 0
        self.epoch = 0

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Trainer initialized. Model: {self.model.model_config}, {n_params:,} trainable parameters, "
                    f"known speakers: {self.model.known_speakers}")
        logger.info(f"Training on {len(train_loader)} batches, validating on {len(val_loader)} batches; "
                    f"device {Config.DEVICE}, mixed precision {'on' if self.use_amp else 'off'}; "
                    f"AdamW lr {Config.LEARNING_RATE} (warmup {warmup} steps, cosine to {Config.MIN_LR_RATIO}x)")
        logger.info(f"Checkpoints and metrics will be written to: {os.path.abspath(self.checkpoint_dir)}")

    # ------------------------------------------------------------------ epochs
    def train_epoch(self):
        self.model.train()
        sums, n = {k: 0.0 for k in self.LOSS_KEYS}, 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        for batch_idx, batch in enumerate(progress_bar):
            d = batch_to_device(batch, Config.DEVICE)
            speaker_ids = self.model.speaker_ids_for(batch['subjects'], Config.DEVICE)
            with autocast_context(Config.DEVICE, self.use_amp):
                predictions = self.model(d['frames'], lengths=d['lengths'], speaker_ids=speaker_ids,
                                         quality=d['quality'])
                losses = self.model.compute_loss(predictions, d['targets'])

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(losses['total_loss']).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=Config.GRAD_CLIP_NORM)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            for k, v in losses.items():
                sums[k] += float(v)
            n += 1
            progress_bar.set_postfix({'loss': f"{float(losses['total_loss']):.3f}",
                                      'ctc': f"{float(losses['ctc_loss']):.3f}",
                                      'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"})
            if batch_idx % 100 == 0:
                step = self.epoch * len(self.train_loader) + batch_idx
                for k, v in losses.items():
                    self.writer.add_scalar(f'Train/{k}', float(v), step)
                self.writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'], step)

        avg = {k: v / max(n, 1) for k, v in sums.items() if v or k in ('total_loss', 'ctc_loss')}
        self.train_losses.append(avg['total_loss'])
        return avg

    def validate_epoch(self):
        result = run_evaluation(self.model, self.val_loader, Config.DEVICE, full=False, amp=self.use_amp,
                                progress=lambda it: tqdm(it, desc="Validation"))
        losses, frame, greedy = result['losses'], result['frame'], result['greedy']
        self.val_losses.append(losses['total_loss'])
        for k, v in losses.items():
            self.writer.add_scalar(f'Val/{k}', v, self.epoch)
        for k in ('wer', 'cer', 'sentence_accuracy'):
            self.writer.add_scalar(f'Val/{k}', greedy[k], self.epoch)
        logger.info(f"Validation - Loss: {losses['total_loss']:.4f} (ctc {losses['ctc_loss']:.4f}) | "
                    f"WER {greedy['wer']:.4f}, CER {greedy['cer']:.4f}, sentence acc {greedy['sentence_accuracy']:.4f} | "
                    f"speech-only Viseme Acc: {frame['speech_viseme_accuracy']:.4f}, "
                    f"Phoneme Acc: {frame['speech_phoneme_accuracy']:.4f}")
        return result

    # ------------------------------------------------------------------ loop
    def _write_metrics_row(self, row):
        write_header = not os.path.exists(self.metrics_path)
        with open(self.metrics_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.METRIC_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, '') for k in self.METRIC_FIELDS})

    def train(self, num_epochs=None):
        num_epochs = num_epochs or self.num_epochs
        start_epoch = self.epoch + 1 if self.resumed else 0
        if not self.resumed and os.path.exists(self.metrics_path):
            os.remove(self.metrics_path)
        logger.info(f"Starting training for epochs {start_epoch + 1}..{num_epochs}")

        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch
            started = time.time()
            train_losses = self.train_epoch()
            logger.info(f"Epoch {epoch + 1} - Training Loss: {train_losses['total_loss']:.4f} "
                        f"(ctc {train_losses['ctc_loss']:.4f})")
            val = self.validate_epoch()

            key = (val['greedy']['wer'], val['losses']['ctc_loss'])
            improved = key < self.best_key
            row = {'epoch': epoch + 1, 'lr': self.optimizer.param_groups[0]['lr'],
                   'epoch_seconds': round(time.time() - started, 1), 'best': int(improved)}
            row.update({f'train_{k}': v for k, v in train_losses.items()})
            row.update({f'val_{k}': v for k, v in val['losses'].items()})
            row.update({'val_wer': val['greedy']['wer'], 'val_cer': val['greedy']['cer'],
                        'val_sentence_accuracy': val['greedy']['sentence_accuracy']})
            row.update({f'val_{k}': val['frame'][k] for k in ('viseme_accuracy', 'phoneme_accuracy',
                                                             'speech_viseme_accuracy', 'speech_phoneme_accuracy')})
            self._write_metrics_row(row)

            if improved:
                self.best_key = key
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_model.pth')
                logger.info(f"New best model saved: val WER {key[0]:.4f}, val CTC loss {key[1]:.4f}")
            else:
                self.epochs_without_improvement += 1
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
            self.save_checkpoint('last_model.pth')

            if self.epochs_without_improvement >= Config.PATIENCE:
                logger.info(f"Early stopping at epoch {epoch + 1}: no validation improvement for {Config.PATIENCE} epochs")
                break

        logger.info("Training completed!")
        if self.test_loader is not None:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            if os.path.exists(best_path):
                best = torch.load(best_path, map_location=Config.DEVICE, weights_only=False)
                self.model.load_state_dict(best['model_state_dict'])
                self.epoch = best['epoch']
                logger.info(f"Loaded best model (epoch {self.epoch + 1}) for test evaluation")
            self.evaluate()
        self.writer.close()

    def evaluate(self):
        logger.info("Evaluating on test set (greedy + grammar-constrained decoding)...")
        result = run_evaluation(self.model, self.test_loader, Config.DEVICE, full=True, amp=self.use_amp,
                                progress=lambda it: tqdm(it, desc="Testing"))
        g, c, f = result['greedy'], result['constrained'], result['frame']
        logger.info("=" * 60)
        logger.info("TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Test Loss: {result['losses']['total_loss']:.4f} (ctc {result['losses']['ctc_loss']:.4f})")
        logger.info(f"Greedy      WER {g['wer']:.4f}  CER {g['cer']:.4f}  sentence acc {g['sentence_accuracy']:.4f}")
        logger.info(f"Constrained WER {c['wer']:.4f}  CER {c['cer']:.4f}  sentence acc {c['sentence_accuracy']:.4f}")
        logger.info(f"Constrained letter slot: acc {c['letter_slot']['accuracy']:.4f}, homophene letters "
                    f"{c['letter_slot']['homophene_letter_accuracy']:.4f}, within-group confusion "
                    f"{c['letter_slot']['within_group_confusion_rate']:.4f}")
        logger.info(f"Frame: viseme {f['viseme_accuracy']:.4f}, phoneme {f['phoneme_accuracy']:.4f}, speech-only "
                    f"viseme {f['speech_viseme_accuracy']:.4f}, phoneme {f['speech_phoneme_accuracy']:.4f}")
        if result['gate']:
            logger.info("Gate (mean language weight): " + ", ".join(
                f"{k} {v['mean_language_weight']:.3f} (n={v['frames']})" for k, v in result['gate'].items()))
        logger.info("=" * 60)

        out = {k: v for k, v in result.items() if not k.startswith('_')}
        out.update({'epoch': self.epoch + 1, 'model_config': self.model.model_config, 'run_info': self.run_info})
        with open(self.test_results_path, 'w') as fh:
            json.dump(out, fh, indent=2, default=float)
        logger.info(f"Test results written to {self.test_results_path}")
        return out

    # ------------------------------------------------------------------ checkpoints
    @staticmethod
    def _config_snapshot():
        simple = (str, int, float, bool, list, tuple, dict, type(None))
        return {k: v for k, v in vars(Config).items() if not k.startswith('__') and isinstance(v, simple)}

    def save_checkpoint(self, filename):
        torch.save({
            'epoch': self.epoch,
            'model_config': self.model.model_config,
            'model_type': self.model.model_type,
            'known_speakers': self.model.known_speakers,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_key': self.best_key,
            'epochs_without_improvement': self.epochs_without_improvement,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'run_info': self.run_info,
            'config': self._config_snapshot(),
        }, os.path.join(self.checkpoint_dir, filename))

    def load_checkpoint(self, filepath):
        """Load a checkpoint to resume training"""
        checkpoint = torch.load(filepath, map_location=Config.DEVICE, weights_only=False)
        saved = checkpoint.get('model_config')
        if saved and saved != self.model.model_config:
            raise ValueError(f"Checkpoint model {saved} differs from the requested model {self.model.model_config}; "
                             f"pass the same --model-type/--seq-model/--visual-pool/--gate to resume.")
        self.model.load_state_dict(upgrade_legacy_state_dict(checkpoint['model_state_dict']))
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_key = tuple(checkpoint.get('best_key', self.best_key))
        self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.resumed = True
        logger.info(f"Checkpoint loaded from epoch {self.epoch + 1} (best val WER so far {self.best_key[0]:.4f})")


def main():
    train_loader, val_loader, test_loader = create_data_loaders(['s1'], ['s2'], ['s3'])
    trainer = LipReadingTrainer(ModelFactory.create_full_model(), train_loader, val_loader, test_loader)
    viseme_mapper.print_mappings()
    trainer.train()


if __name__ == "__main__":
    main()
