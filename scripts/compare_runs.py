"""Compare finished runs side by side: one row per run folder that contains test results.

Reads, from each <runs-dir>/<run>/:
  evaluation/evaluation_results.json            clean test metrics (falls back to test_results.json)
  evaluation/evaluation_results_corrupt*.json   robustness test (optional)
  training_metrics.csv                          epochs, time, best validation WER
  best_model.pth                                parameter count and model configuration

Writes comparison.csv, comparison.md and comparison.png into <runs-dir> (or --out) and prints the table.
Lower WER / CER is better. "robust dWER" is the WER increase when ~30% of the test frames are corrupted.

Usage:  python scripts/compare_runs.py checkpoints/abl-seq [more run folders or groups ...] [--out DIR]
"""
import argparse
import csv
import glob
import json
import os

COLUMNS = [
    ('run', 'Run'), ('model_type', 'Model'), ('seq_model', 'Seq'), ('visual_pool', 'Pool'), ('gate', 'Gate'),
    ('params_m', 'Params (M)'), ('epochs', 'Epochs'), ('best_epoch', 'Best ep.'), ('min_per_epoch', 'Min/epoch'),
    ('val_wer', 'Val WER'), ('greedy_wer', 'Greedy WER'), ('greedy_cer', 'Greedy CER'),
    ('wer', 'Constr. WER'), ('cer', 'Constr. CER'), ('sentence_acc', 'Sentence acc'),
    ('letter_acc', 'Letter acc'), ('homophene_letter_acc', 'Homophene-letter acc'),
    ('speech_phoneme_acc', 'Speech phoneme acc'), ('corrupt_wer', 'WER (corrupted)'), ('robust_dwer', 'Robust dWER'),
    ('lang_w_reliable', 'LM weight reliable'), ('lang_w_unreliable', 'LM weight unreliable'),
]


def find_runs(paths):
    runs = []
    for path in paths:
        candidates = [path] + sorted(glob.glob(os.path.join(path, '*/')))
        for d in candidates:
            d = d.rstrip('/')
            if os.path.exists(os.path.join(d, 'evaluation', 'evaluation_results.json')) or \
                    os.path.exists(os.path.join(d, 'test_results.json')):
                runs.append(d)
    return list(dict.fromkeys(runs))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def summarise(run_dir):
    eval_path = os.path.join(run_dir, 'evaluation', 'evaluation_results.json')
    res = load_json(eval_path) if os.path.exists(eval_path) else load_json(os.path.join(run_dir, 'test_results.json'))
    row = {'run': os.path.basename(run_dir)}
    row.update(res.get('model_config') or {})
    if row.get('model_type') != 'full':
        row['gate'] = '-'

    best = os.path.join(run_dir, 'best_model.pth')
    if os.path.exists(best):
        try:
            import torch
            ckpt = torch.load(best, map_location='cpu', weights_only=False)
            row['params_m'] = sum(v.numel() for k, v in ckpt['model_state_dict'].items()
                                  if v.is_floating_point() and not k.endswith(('running_mean', 'running_var',
                                                                              'confusion_matrix', 'ambiguity_table',
                                                                              'pe'))) / 1e6
            row['best_epoch'] = ckpt.get('epoch', -1) + 1
        except Exception as e:  # comparison still works without the checkpoint
            print(f"[{row['run']}] could not read best_model.pth: {e}")

    metrics = os.path.join(run_dir, 'training_metrics.csv')
    if os.path.exists(metrics):
        rows = list(csv.DictReader(open(metrics)))
        if rows:
            row['epochs'] = len(rows)
            secs = sorted(float(r['epoch_seconds']) for r in rows)
            row['min_per_epoch'] = secs[len(secs) // 2] / 60  # median: robust to a stalled epoch
            if 'val_wer' in rows[0]:
                row['val_wer'] = min(float(r['val_wer']) for r in rows)

    g, c = res['greedy'], res.get('constrained', res['greedy'])
    row.update({'greedy_wer': g['wer'], 'greedy_cer': g['cer'], 'wer': c['wer'], 'cer': c['cer'],
                'sentence_acc': c['sentence_accuracy'], 'letter_acc': c['letter_slot']['accuracy'],
                'homophene_letter_acc': c['letter_slot']['homophene_letter_accuracy'],
                'speech_phoneme_acc': res['frame']['speech_phoneme_accuracy']})

    corrupt = sorted(glob.glob(os.path.join(run_dir, 'evaluation', 'evaluation_results_corrupt*.json')))
    if corrupt:
        cr = load_json(corrupt[0])
        cc = cr.get('constrained', cr['greedy'])
        row['corrupt_wer'] = cc['wer']
        row['robust_dwer'] = cc['wer'] - c['wer']
        gate = cr.get('gate') or {}
        if gate:
            def mean_w(keys):
                n = sum(gate[k]['frames'] for k in keys)
                return sum(gate[k]['mean_language_weight'] * gate[k]['frames'] for k in keys) / n if n else None
            row['lang_w_reliable'] = mean_w(['clear_reliable', 'ambiguous_reliable'])
            row['lang_w_unreliable'] = mean_w(['clear_unreliable', 'ambiguous_unreliable'])
    return row


def fmt(v):
    if v is None or v == '':
        return '-'
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}"
    return str(v)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+', help='Run folders, or folders containing run folders')
    parser.add_argument('--out', help='Where to write comparison.csv/.md/.png (default: the first path)')
    args = parser.parse_args()

    runs = find_runs(args.paths)
    if not runs:
        raise SystemExit(f"No finished runs (test_results.json / evaluation) found under {args.paths}")
    rows = sorted((summarise(r) for r in runs), key=lambda r: (r.get('wer', 9), r.get('cer', 9)))
    out = args.out or args.paths[0]
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(out, 'comparison.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[k for k, _ in COLUMNS], extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    shown = [(k, h) for k, h in COLUMNS if any(r.get(k) not in (None, '') for r in rows)]
    md = ['| ' + ' | '.join(h for _, h in shown) + ' |', '|' + '---|' * len(shown)]
    md += ['| ' + ' | '.join(fmt(r.get(k)) for k, _ in shown) + ' |' for r in rows]
    note = ("\nSorted by grammar-constrained WER on the test speakers (lower is better). Constr. = GRID-grammar "
            "decoding; Robust dWER = WER increase with ~30% of frames corrupted; LM weight = mean language-model "
            "weight of the fusion gate on reliable vs unreliable frames in the corrupted test.\n")
    with open(os.path.join(out, 'comparison.md'), 'w') as f:
        f.write('\n'.join(md) + '\n' + note)
    print('\n'.join(md))
    print(note)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        names = [r['run'] for r in rows]
        x = np.arange(len(rows))
        series = [('wer', 'WER (constrained)'), ('cer', 'CER (constrained)'), ('greedy_wer', 'WER (greedy)'),
                  ('corrupt_wer', 'WER (30% frames corrupted)')]
        series = [(k, l) for k, l in series if all(r.get(k) is not None for r in rows)]
        width = 0.8 / len(series)
        fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(rows) + 3), 5))
        for i, (k, label) in enumerate(series):
            vals = [r[k] for r in rows]
            bars = ax.bar(x + (i - (len(series) - 1) / 2) * width, vals, width, label=label)
            ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], fontsize=7, padding=2)
        ax.set_xticks(x, names, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel('error rate (lower is better)')
        ax.set_ylim(0, min(1.15, max(r.get(k, 0) for r in rows for k, _ in series) * 1.2 + 0.02))
        ax.set_title('Test error rates by configuration')
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=len(series), frameon=False)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out, 'comparison.png'), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"(plot skipped: {e})")
    print(f"Wrote {os.path.join(out, 'comparison.csv')}, comparison.md and comparison.png")


if __name__ == '__main__':
    main()
