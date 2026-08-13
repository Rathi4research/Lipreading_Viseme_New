import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from src.config import Config

logger = logging.getLogger(__name__)


class LipReadingEvaluator:
    """Enhanced evaluation for lip reading model with comprehensive analysis"""

    def __init__(self, model, test_loader, device='cuda'):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.results = {}
        # 'visual_only' skips the language-model branch, which otherwise receives the
        # ground-truth phoneme sequence as input (models/fusion_network.py forward()) —
        # using 'full' here would let the model echo the answer key back for phoneme_logits.
        self.eval_model_type = 'visual_only'

    def evaluate_comprehensive(self):
        """Run comprehensive evaluation with adaptive thresholds"""
        logger.info("Starting comprehensive evaluation...")

        self.model.eval()

        # Accumulators
        total_samples = 0
        correct_visemes = 0
        correct_phonemes = 0

        # Store all data for post-processing
        all_viseme_preds = []
        all_viseme_targets = []
        all_phoneme_preds = []
        all_phoneme_targets = []
        all_ambiguity_scores = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                if batch_idx % 10 == 0:
                    logger.info(f"Processing batch {batch_idx}/{len(self.test_loader)}")

                frames = batch['frames'].to(self.device)
                visemes = batch['visemes'].to(self.device)
                phonemes = batch['phonemes'].to(self.device)
                ambiguity_scores = batch['ambiguity_scores'].to(self.device)

                # Speaker mapping
                speaker_ids = torch.tensor([
                    Config.SPEAKER_TO_ID.get(s, 0)
                    for s in batch['subjects']
                ]).to(self.device)

                # Forward pass — model_type='visual_only' so the language-model branch
                # (which otherwise takes the ground-truth phoneme sequence as input) is
                # skipped, giving an honest measure of what the visual pipeline learned.
                predictions = self.model(
                    frames=frames,
                    phoneme_targets=phonemes,
                    speaker_ids=speaker_ids,
                    ambiguity_scores=ambiguity_scores,
                    model_type=self.eval_model_type
                )

                # Get predictions
                viseme_preds = torch.argmax(predictions['viseme_logits'], dim=-1)
                phoneme_preds = torch.argmax(predictions['phoneme_logits'], dim=-1)

                # Flatten
                viseme_preds_flat = viseme_preds.view(-1)
                viseme_targets_flat = visemes.view(-1)
                phoneme_preds_flat = phoneme_preds.view(-1)
                phoneme_targets_flat = phonemes.view(-1)
                ambiguity_flat = ambiguity_scores.view(-1)

                # Valid samples (not padding)
                valid_mask = (viseme_targets_flat != -1) & (phoneme_targets_flat != -1)

                if valid_mask.sum() == 0:
                    continue

                # Apply mask
                v_preds = viseme_preds_flat[valid_mask]
                v_targets = viseme_targets_flat[valid_mask]
                p_preds = phoneme_preds_flat[valid_mask]
                p_targets = phoneme_targets_flat[valid_mask]
                amb_scores = ambiguity_flat[valid_mask]

                # Store data
                all_viseme_preds.extend(v_preds.cpu().numpy())
                all_viseme_targets.extend(v_targets.cpu().numpy())
                all_phoneme_preds.extend(p_preds.cpu().numpy())
                all_phoneme_targets.extend(p_targets.cpu().numpy())
                all_ambiguity_scores.extend(amb_scores.cpu().numpy())

                # Update counters
                batch_samples = len(v_targets)
                total_samples += batch_samples
                correct_visemes += (v_preds == v_targets).sum().item()
                correct_phonemes += (p_preds == p_targets).sum().item()

        # Convert to numpy arrays
        all_ambiguity_scores = np.array(all_ambiguity_scores)
        all_phoneme_preds = np.array(all_phoneme_preds)
        all_phoneme_targets = np.array(all_phoneme_targets)
        all_viseme_preds = np.array(all_viseme_preds)
        all_viseme_targets = np.array(all_viseme_targets)

        # Determine adaptive thresholds based on data distribution
        percentile_25 = np.percentile(all_ambiguity_scores, 25)
        percentile_75 = np.percentile(all_ambiguity_scores, 75)

        print(f"\n{'=' * 70}")
        print(f"AMBIGUITY DISTRIBUTION ANALYSIS")
        print(f"{'=' * 70}")
        print(f"Min: {all_ambiguity_scores.min():.4f}")
        print(f"25th percentile: {percentile_25:.4f}")
        print(f"Median: {np.median(all_ambiguity_scores):.4f}")
        print(f"75th percentile: {percentile_75:.4f}")
        print(f"Max: {all_ambiguity_scores.max():.4f}")
        print(f"Mean: {all_ambiguity_scores.mean():.4f} ± {all_ambiguity_scores.std():.4f}")

        # Use adaptive thresholds
        low_threshold = percentile_25
        high_threshold = percentile_75

        # Calculate ambiguity-based metrics
        high_amb_mask = all_ambiguity_scores >= high_threshold
        low_amb_mask = all_ambiguity_scores <= low_threshold
        mid_amb_mask = ~high_amb_mask & ~low_amb_mask

        high_amb_total = high_amb_mask.sum()
        low_amb_total = low_amb_mask.sum()
        mid_amb_total = mid_amb_mask.sum()

        high_amb_correct = (all_phoneme_preds[high_amb_mask] == all_phoneme_targets[high_amb_mask]).sum()
        low_amb_correct = (all_phoneme_preds[low_amb_mask] == all_phoneme_targets[low_amb_mask]).sum()
        mid_amb_correct = (all_phoneme_preds[mid_amb_mask] == all_phoneme_targets[mid_amb_mask]).sum()

        print(f"\nUsing adaptive thresholds:")
        print(
            f"  Low ambiguity: ≤ {low_threshold:.4f} ({low_amb_total} samples, {100 * low_amb_total / total_samples:.1f}%)")
        print(
            f"  Mid ambiguity: {low_threshold:.4f} - {high_threshold:.4f} ({mid_amb_total} samples, {100 * mid_amb_total / total_samples:.1f}%)")
        print(
            f"  High ambiguity: ≥ {high_threshold:.4f} ({high_amb_total} samples, {100 * high_amb_total / total_samples:.1f}%)")

        # Generate confusion matrices
        viseme_cm = confusion_matrix(all_viseme_targets, all_viseme_preds)
        phoneme_cm = confusion_matrix(all_phoneme_targets, all_phoneme_preds)

        # Calculate per-class accuracies. A row sum of 0 means that class never appeared
        # in the targets, so accuracy is undefined (not 0) — leave those classes as nan
        # and exclude them from aggregates like min/max instead of dividing by zero.
        viseme_row_sums = viseme_cm.sum(axis=1)
        viseme_per_class_acc = np.divide(
            viseme_cm.diagonal(), viseme_row_sums,
            out=np.full(viseme_row_sums.shape, np.nan), where=viseme_row_sums != 0
        )
        phoneme_row_sums = phoneme_cm.sum(axis=1)
        phoneme_per_class_acc = np.divide(
            phoneme_cm.diagonal(), phoneme_row_sums,
            out=np.full(phoneme_row_sums.shape, np.nan), where=phoneme_row_sums != 0
        )

        results = {
            'accuracy_metrics': {
                'viseme_accuracy': correct_visemes / total_samples if total_samples > 0 else 0.0,
                'phoneme_accuracy': correct_phonemes / total_samples if total_samples > 0 else 0.0,
                'total_samples': total_samples
            },
            'ambiguity_analysis': {
                'high_ambiguity_accuracy': high_amb_correct / high_amb_total if high_amb_total > 0 else 0.0,
                'mid_ambiguity_accuracy': mid_amb_correct / mid_amb_total if mid_amb_total > 0 else 0.0,
                'low_ambiguity_accuracy': low_amb_correct / low_amb_total if low_amb_total > 0 else 0.0,
                'high_ambiguity_samples': int(high_amb_total),
                'mid_ambiguity_samples': int(mid_amb_total),
                'low_ambiguity_samples': int(low_amb_total),
                'thresholds': {
                    'low': float(low_threshold),
                    'high': float(high_threshold)
                }
            },
            'confusion_matrices': {
                'viseme_confusion_matrix': viseme_cm,
                'phoneme_confusion_matrix': phoneme_cm,
                'viseme_per_class_accuracy': viseme_per_class_acc,
                'phoneme_per_class_accuracy': phoneme_per_class_acc
            },
            'detailed_analysis': {
                'ambiguity_distribution': {
                    'mean': float(np.mean(all_ambiguity_scores)),
                    'std': float(np.std(all_ambiguity_scores)),
                    'min': float(np.min(all_ambiguity_scores)),
                    'max': float(np.max(all_ambiguity_scores)),
                    'percentiles': {
                        '10': float(np.percentile(all_ambiguity_scores, 10)),
                        '25': float(percentile_25),
                        '50': float(np.percentile(all_ambiguity_scores, 50)),
                        '75': float(percentile_75),
                        '90': float(np.percentile(all_ambiguity_scores, 90))
                    }
                }
            }
        }

        self.results = results

        print(f"\n{'=' * 70}")
        print(f"PERFORMANCE SUMMARY")
        print(f"{'=' * 70}")
        print(f"Overall Viseme Accuracy: {results['accuracy_metrics']['viseme_accuracy']:.4f}")
        print(f"Overall Phoneme Accuracy: {results['accuracy_metrics']['phoneme_accuracy']:.4f}")
        print(f"\nAccuracy by Ambiguity Level:")
        print(f"  Low  ({low_amb_total:6d} samples): {results['ambiguity_analysis']['low_ambiguity_accuracy']:.4f}")
        print(f"  Mid  ({mid_amb_total:6d} samples): {results['ambiguity_analysis']['mid_ambiguity_accuracy']:.4f}")
        print(f"  High ({high_amb_total:6d} samples): {results['ambiguity_analysis']['high_ambiguity_accuracy']:.4f}")
        print(f"{'=' * 70}\n")

        logger.info("Evaluation completed successfully")
        return results

    def plot_confusion_matrix(self, cm, labels, title, save_path=None, figsize=(12, 10)):
        """Plot confusion matrix with better visualization"""

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Normalize confusion matrix for better visualization. Rows with zero true
        # samples have no meaningful frequency distribution, so leave them as 0 rather
        # than dividing by zero.
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.divide(
            cm.astype('float'), row_sums,
            out=np.zeros_like(cm, dtype=float), where=row_sums != 0
        )

        # Plot heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Normalized Frequency'},
                    ax=ax, vmin=0, vmax=1)

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')

        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
            print(f"Confusion matrix saved to: {save_path}")

        plt.close()

    def plot_ambiguity_analysis(self, save_path='plots/ambiguity_analysis.png'):
        """Plot comprehensive ambiguity analysis"""
        if not self.results:
            logger.error("No results available. Run evaluate_comprehensive() first.")
            return

        amb_dist = self.results['detailed_analysis']['ambiguity_distribution']
        amb_analysis = self.results['ambiguity_analysis']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Histogram of ambiguity scores
        ax1 = axes[0, 0]
        # Create synthetic data based on distribution (since we don't store raw scores)
        # This is for visualization purposes
        ax1.axvline(amb_analysis['thresholds']['low'], color='g', linestyle='--',
                    label=f"Low threshold ({amb_analysis['thresholds']['low']:.3f})", linewidth=2)
        ax1.axvline(amb_analysis['thresholds']['high'], color='r', linestyle='--',
                    label=f"High threshold ({amb_analysis['thresholds']['high']:.3f})", linewidth=2)
        ax1.set_xlabel('Ambiguity Score', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('Ambiguity Score Distribution', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.text(0.02, 0.98, f"Mean: {amb_dist['mean']:.3f}\nStd: {amb_dist['std']:.3f}",
                 transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 2: Accuracy by ambiguity level
        ax2 = axes[0, 1]
        categories = ['Low\nAmbiguity', 'Mid\nAmbiguity', 'High\nAmbiguity']
        accuracies = [
            amb_analysis['low_ambiguity_accuracy'],
            amb_analysis['mid_ambiguity_accuracy'],
            amb_analysis['high_ambiguity_accuracy']
        ]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        bars = ax2.bar(categories, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Phoneme Accuracy', fontsize=11, fontweight='bold')
        ax2.set_title('Accuracy by Ambiguity Level', fontsize=13, fontweight='bold')
        ax2.set_ylim([0, 1.0])
        ax2.grid(True, axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        # Plot 3: Sample distribution
        ax3 = axes[1, 0]
        samples = [
            amb_analysis['low_ambiguity_samples'],
            amb_analysis['mid_ambiguity_samples'],
            amb_analysis['high_ambiguity_samples']
        ]
        total = sum(samples)
        percentages = [s / total * 100 for s in samples]

        wedges, texts, autotexts = ax3.pie(samples, labels=categories, autopct='%1.1f%%',
                                           colors=colors, startangle=90,
                                           textprops={'fontsize': 10, 'fontweight': 'bold'})
        ax3.set_title('Sample Distribution by Ambiguity', fontsize=13, fontweight='bold')

        # Add legend with counts
        legend_labels = [f"{cat}: {samp:,} samples" for cat, samp in zip(categories, samples)]
        ax3.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1))

        # Plot 4: Performance summary table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')

        table_data = [
            ['Metric', 'Value'],
            ['', ''],
            ['Overall Viseme Accuracy', f"{self.results['accuracy_metrics']['viseme_accuracy']:.4f}"],
            ['Overall Phoneme Accuracy', f"{self.results['accuracy_metrics']['phoneme_accuracy']:.4f}"],
            ['Total Samples', f"{self.results['accuracy_metrics']['total_samples']:,}"],
            ['', ''],
            ['Low Amb. Accuracy', f"{amb_analysis['low_ambiguity_accuracy']:.4f}"],
            ['Mid Amb. Accuracy', f"{amb_analysis['mid_ambiguity_accuracy']:.4f}"],
            ['High Amb. Accuracy', f"{amb_analysis['high_ambiguity_accuracy']:.4f}"],
        ]

        table = ax4.table(cellText=table_data, cellLoc='left', loc='center',
                          colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style data rows
        for i in range(2, len(table_data)):
            for j in range(2):
                if i == 5:  # Empty row
                    continue
                table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')

        ax4.set_title('Performance Summary', fontsize=13, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Ambiguity analysis saved to {save_path}")
            print(f"Ambiguity analysis plot saved to: {save_path}")

        plt.close()

    def generate_report(self, save_path='results/evaluation_report.txt'):
        """Generate comprehensive evaluation report"""
        if not self.results:
            logger.error("No evaluation results available. Run evaluate_comprehensive() first.")
            return

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ENHANCED VISEME-PHONEME ADAPTIVE MODEL - EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Basic metrics
        acc_metrics = self.results['accuracy_metrics']
        report_lines.append("OVERALL PERFORMANCE")
        report_lines.append("-" * 40)
        report_lines.append(f"Viseme Accuracy: {acc_metrics['viseme_accuracy']:.4f}")
        report_lines.append(f"Phoneme Accuracy: {acc_metrics['phoneme_accuracy']:.4f}")
        report_lines.append(f"Total Test Samples: {acc_metrics['total_samples']:,}")
        report_lines.append("")

        # Ambiguity analysis
        amb_analysis = self.results['ambiguity_analysis']
        report_lines.append("AMBIGUITY-BASED PERFORMANCE")
        report_lines.append("-" * 40)
        report_lines.append(f"Adaptive Thresholds Used:")
        report_lines.append(f"  Low threshold: ≤ {amb_analysis['thresholds']['low']:.4f}")
        report_lines.append(f"  High threshold: ≥ {amb_analysis['thresholds']['high']:.4f}")
        report_lines.append("")
        report_lines.append(
            f"Low Ambiguity Accuracy: {amb_analysis['low_ambiguity_accuracy']:.4f} ({amb_analysis['low_ambiguity_samples']:,} samples)")
        report_lines.append(
            f"Mid Ambiguity Accuracy: {amb_analysis['mid_ambiguity_accuracy']:.4f} ({amb_analysis['mid_ambiguity_samples']:,} samples)")
        report_lines.append(
            f"High Ambiguity Accuracy: {amb_analysis['high_ambiguity_accuracy']:.4f} ({amb_analysis['high_ambiguity_samples']:,} samples)")
        report_lines.append("")

        # Ambiguity distribution
        amb_dist = self.results['detailed_analysis']['ambiguity_distribution']
        report_lines.append("AMBIGUITY SCORE DISTRIBUTION")
        report_lines.append("-" * 40)
        report_lines.append(f"Mean: {amb_dist['mean']:.4f}")
        report_lines.append(f"Std Dev: {amb_dist['std']:.4f}")
        report_lines.append(f"Range: {amb_dist['min']:.4f} - {amb_dist['max']:.4f}")
        report_lines.append(f"Percentiles:")
        report_lines.append(f"  10th: {amb_dist['percentiles']['10']:.4f}")
        report_lines.append(f"  25th: {amb_dist['percentiles']['25']:.4f}")
        report_lines.append(f"  50th (Median): {amb_dist['percentiles']['50']:.4f}")
        report_lines.append(f"  75th: {amb_dist['percentiles']['75']:.4f}")
        report_lines.append(f"  90th: {amb_dist['percentiles']['90']:.4f}")
        report_lines.append("")

        # Confusion matrix summary
        if 'confusion_matrices' in self.results:
            v_cm = self.results['confusion_matrices']['viseme_confusion_matrix']
            p_cm = self.results['confusion_matrices']['phoneme_confusion_matrix']
            v_per_class = self.results['confusion_matrices']['viseme_per_class_accuracy']
            p_per_class = self.results['confusion_matrices']['phoneme_per_class_accuracy']

            report_lines.append("CONFUSION MATRIX SUMMARY")
            report_lines.append("-" * 40)
            report_lines.append(f"Viseme Classes: {v_cm.shape[0]}")
            report_lines.append(f"Phoneme Classes: {p_cm.shape[0]}")
            report_lines.append("")
            report_lines.append("Per-Class Performance:")
            report_lines.append(f"  Viseme - Worst class accuracy: {np.nanmin(v_per_class):.4f}")
            report_lines.append(f"  Viseme - Best class accuracy: {np.nanmax(v_per_class):.4f}")
            report_lines.append(f"  Phoneme - Worst class accuracy: {np.nanmin(p_per_class):.4f}")
            report_lines.append(f"  Phoneme - Best class accuracy: {np.nanmax(p_per_class):.4f}")
            report_lines.append("")

        report_lines.append("=" * 80)

        # Save report
        report_text = "\n".join(report_lines)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        print(report_text)
        logger.info(f"Evaluation report saved to {save_path}")

        return report_text

    def quick_report(self, results):
        """Generate a quick summary report"""
        print("=" * 60)
        print("QUICK EVALUATION REPORT")
        print("=" * 60)

        if 'error' in results:
            print(f"Error: {results['error']}")
            return

        if 'accuracy_metrics' in results:
            acc = results['accuracy_metrics']
            amb = results['ambiguity_analysis']
            print(f"Viseme Accuracy: {acc['viseme_accuracy']:.4f}")
            print(f"Phoneme Accuracy: {acc['phoneme_accuracy']:.4f}")
            print(f"\nAmbiguity-based Performance:")
            print(f"  Low:  {amb['low_ambiguity_accuracy']:.4f} ({amb['low_ambiguity_samples']:,} samples)")
            print(f"  Mid:  {amb['mid_ambiguity_accuracy']:.4f} ({amb['mid_ambiguity_samples']:,} samples)")
            print(f"  High: {amb['high_ambiguity_accuracy']:.4f} ({amb['high_ambiguity_samples']:,} samples)")
            print(f"Total Samples: {acc['total_samples']:,}")

        print("=" * 60)