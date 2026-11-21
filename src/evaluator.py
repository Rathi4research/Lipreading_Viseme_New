"""Stand-alone evaluation (main.py evaluate): metrics, confusion-matrix plots, text report and JSON."""
import json
import logging
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import Config
from src.metrics import run_evaluation
from src.viseme_mapping import viseme_mapper

logger = logging.getLogger(__name__)


class LipReadingEvaluator:
    def __init__(self, model, test_loader, device='cuda'):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.results = {}

    def evaluate_comprehensive(self):
        logger.info(f"Evaluating model {self.model.model_config} on {len(self.test_loader.dataset)} clips...")
        r = run_evaluation(self.model, self.test_loader, self.device, full=True,
                           amp=Config.AMP and str(self.device).startswith('cuda'))
        self.results = r
        g, c, f = r['greedy'], r['constrained'], r['frame']
        print(f"\n{'=' * 70}\nEVALUATION SUMMARY\n{'=' * 70}")
        print(f"Greedy      WER {g['wer']:.4f}  CER {g['cer']:.4f}  sentence acc {g['sentence_accuracy']:.4f}")
        print(f"Constrained WER {c['wer']:.4f}  CER {c['cer']:.4f}  sentence acc {c['sentence_accuracy']:.4f}")
        ls = c['letter_slot']
        print(f"Letter slot (constrained): acc {ls['accuracy']:.4f} | homophene letters {ls['homophene_letter_accuracy']:.4f} "
              f"| distinct letters {ls['distinct_letter_accuracy']:.4f} | within-group confusion {ls['within_group_confusion_rate']:.4f}")
        print(f"Frame: viseme {f['viseme_accuracy']:.4f}, phoneme {f['phoneme_accuracy']:.4f} | speech-only viseme "
              f"{f['speech_viseme_accuracy']:.4f}, phoneme {f['speech_phoneme_accuracy']:.4f}")
        if r['gate']:
            print("Gate, mean language weight: " + ", ".join(
                f"{k} {v['mean_language_weight']:.3f} (n={v['frames']})" for k, v in r['gate'].items()))
        print(f"{'=' * 70}\n")
        return r

    def plot_confusion_matrix(self, cm, labels, title, save_path=None, figsize=(12, 10)):
        fig, ax = plt.subplots(figsize=figsize)
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        sns.heatmap(cm_norm, annot=len(labels) <= 20, fmt='.2f', cmap='YlOrRd', xticklabels=labels,
                    yticklabels=labels, cbar_kws={'label': 'Normalized Frequency'}, ax=ax, vmin=0, vmax=1)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        plt.close(fig)

    def save_plots(self, plots_dir='plots'):
        r = self.results
        viseme_labels = [f"V{i} " + "/".join(p for p in viseme_mapper.viseme_to_phonemes[i] if p in Config.PHONEMES)[:12]
                         for i in range(viseme_mapper.num_visemes)]
        self.plot_confusion_matrix(r['_viseme_cm'], viseme_labels, "Viseme Confusion Matrix",
                                   os.path.join(plots_dir, 'viseme_confusion_matrix.png'))
        cm = r['_phoneme_cm']
        present = [i for i in range(cm.shape[0]) if cm[i].sum() > 0]
        self.plot_confusion_matrix(cm[np.ix_(present, present)], [Config.PHONEMES[i] for i in present],
                                   "Phoneme Confusion Matrix (phonemes present in test set)",
                                   os.path.join(plots_dir, 'phoneme_confusion_matrix.png'),
                                   figsize=(max(12, len(present) * 0.45), max(10, len(present) * 0.4)))

    def save_json(self, path='results/evaluation_results.json', extra=None):
        out = {k: v for k, v in self.results.items() if not k.startswith('_')}
        out['model_config'] = self.model.model_config
        out.update(extra or {})
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(out, f, indent=2, default=float)
        print(f"Evaluation JSON saved to: {path}")

    def generate_report(self, save_path='results/evaluation_report.txt'):
        r = self.results
        if not r:
            logger.error("No evaluation results available. Run evaluate_comprehensive() first.")
            return
        lines = ["=" * 78, "VISEME-PHONEME ADAPTIVE MODEL - EVALUATION REPORT", "=" * 78,
                 f"Model: {self.model.model_config}", ""]
        for name in ('greedy', 'constrained'):
            s = r[name]
            lines += [f"{name.upper()} DECODING", "-" * 40,
                      f"WER {s['wer']:.4f}   CER {s['cer']:.4f}   sentence accuracy {s['sentence_accuracy']:.4f}   "
                      f"well-formed {s['well_formed_rate']:.4f}",
                      "Slot accuracy: " + ", ".join(f"{k} {v:.3f}" for k, v in s['slot_accuracy'].items())]
            ls = s['letter_slot']
            lines += [f"Letter slot: accuracy {ls['accuracy']:.4f}; homophene letters {ls['homophene_letter_accuracy']:.4f} "
                      f"(n={ls['homophene_letters']}); distinct letters {ls['distinct_letter_accuracy']:.4f}; "
                      f"within-group confusion {ls['within_group_confusion_rate']:.4f}",
                      "Letter pair confusion: " + ", ".join(f"{k} {v:.3f} (n={ls['pair_counts'][k]})"
                                                            for k, v in ls['pair_confusion_rate'].items()), ""]
        lines.append(f"Letter homophene groups (identical viseme sequences): {r['homophene_groups']}")
        f = r['frame']
        lines += ["", "FRAME-LEVEL", "-" * 40,
                  f"Viseme {f['viseme_accuracy']:.4f}  Phoneme {f['phoneme_accuracy']:.4f}  "
                  f"Speech-only viseme {f['speech_viseme_accuracy']:.4f}  phoneme {f['speech_phoneme_accuracy']:.4f}",
                  f"High-ambiguity phoneme {f['high_ambiguity_phoneme_accuracy']:.4f}  "
                  f"Low-ambiguity phoneme {f['low_ambiguity_phoneme_accuracy']:.4f}",
                  "Phoneme accuracy within viseme class: " + ", ".join(
                      f"{k} {v['phoneme_accuracy']:.3f}" for k, v in f['phoneme_accuracy_within_viseme_class'].items())]
        if r['gate']:
            lines += ["", "FUSION GATE (mean language-model weight)", "-" * 40]
            for k, v in r['gate'].items():
                extra = "".join(f", {e} {v[e]:.3f}" for e in ('mean_intrinsic_logit', 'mean_extrinsic_logit') if e in v)
                lines.append(f"{k:22s} {v['mean_language_weight']:.3f} (frames {v['frames']}{extra})")
        lines += ["", "EXAMPLES (reference | greedy | constrained)", "-" * 40]
        lines += [f"{e['reference']} | {e['greedy']} | {e['constrained']}" for e in r['examples'][:10]]
        lines.append("=" * 78)
        text = "\n".join(lines)
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as fh:
            fh.write(text)
        print(text)
        logger.info(f"Evaluation report saved to {save_path}")
        return text
