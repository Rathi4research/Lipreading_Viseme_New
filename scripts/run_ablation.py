"""Run an ablation: train + evaluate several model configurations one after another, with identical data
and training settings, each in its own run folder. Re-running continues where it stopped: finished runs
are skipped and an interrupted run resumes from its last_model.pth.

A configuration is written model_type/seq_model/visual_pool/gate[/lm_context],
e.g. full/bilstm/avg/intrinsic-extrinsic or full/conformer/avg/intrinsic-extrinsic/draft.
Presets:
  seq    full model, intrinsic-extrinsic gate, sequence encoders hybrid | bilstm | transformer | conformer
  gate   hybrid encoder; gates intrinsic-extrinsic | ambiguity | entropy | learned | hard, plus the baseline
         (plain fusion) and visual_only (no language model) model types
  pool   visual pooling avg vs flatten (full / hybrid / intrinsic-extrinsic)

Everything after "--" is passed unchanged to every `main.py train` call (speakers, epochs, batch size, ...).

Examples:
  python scripts/run_ablation.py --preset seq --runs-dir checkpoints/abl-seq -- \\
      --train-subjects s1 --val-subjects s2 --test-subjects s2 --num-epochs 6 --batch-size 8
  python scripts/run_ablation.py --configs full/hybrid/avg/hard baseline/hybrid/avg/ambiguity --runs-dir runs/x -- ...
After the runs, compare them with:  python scripts/compare_runs.py <runs-dir>
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRESETS = {
    'seq': [f'full/{s}/avg/intrinsic-extrinsic' for s in ('hybrid', 'bilstm', 'transformer', 'conformer')],
    'gate': [f'full/hybrid/avg/{g}' for g in ('intrinsic-extrinsic', 'ambiguity', 'entropy', 'learned', 'hard')]
            + ['baseline/hybrid/avg/ambiguity', 'visual_only/hybrid/avg/ambiguity'],
    'pool': ['full/hybrid/avg/intrinsic-extrinsic', 'full/hybrid/flatten/intrinsic-extrinsic'],
}


def parse_config(spec):
    parts = spec.split('/')
    if len(parts) not in (4, 5):
        raise SystemExit(f"Configuration must be model_type/seq_model/visual_pool/gate[/lm_context], got {spec!r}")
    model_type, seq_model, visual_pool, gate = parts[:4]
    lm_context = parts[4] if len(parts) == 5 else 'none'
    # the gate does not affect baseline / visual_only models; name them without it
    name = f"{model_type}-{seq_model}-{visual_pool}" + (f"-{gate}" if model_type == 'full' else '')
    name += f"-{lm_context}" if lm_context != 'none' else ''
    return name, {'model_type': model_type, 'seq_model': seq_model, 'visual_pool': visual_pool,
                  'gate': gate, 'lm_context': lm_context}


def flag_value(args, flag):
    """Value(s) of a flag in a pass-through argument list (None if absent)."""
    if flag not in args:
        return None
    i = args.index(flag) + 1
    values = []
    while i < len(args) and not args[i].startswith('--'):
        values.append(args[i])
        i += 1
    return values


def run(cmd, log_path):
    print(f"$ {' '.join(cmd)}", flush=True)
    with open(log_path, 'a') as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTORCH_ENABLE_MPS_FALLBACK': '1'})
        for line in proc.stdout:
            log.write(line)
            # keep the console short: progress bars and per-batch lines stay in the run log
            if any(k in line for k in ('Validation -', 'New best', 'Early stopping', 'Greedy ', 'Constrained WER',
                                       'Gate (', 'Error', 'Traceback', 'Checkpoint loaded')):
                sys.stdout.write('    ' + line.split(' - INFO - ')[-1])
                sys.stdout.flush()
        return proc.wait()


def main():
    argv = sys.argv[1:]
    train_args = argv[argv.index('--') + 1:] if '--' in argv else []
    own = argv[:argv.index('--')] if '--' in argv else argv

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preset', choices=sorted(PRESETS), help='Predefined set of configurations')
    parser.add_argument('--configs', nargs='+', default=[],
                        help='Configurations model_type/seq_model/visual_pool/gate[/lm_context]')
    parser.add_argument('--runs-dir', required=True, help='Folder that receives one sub-folder per configuration')
    parser.add_argument('--corrupt-ratio', type=float, default=0.3,
                        help='Also evaluate with this share of frames corrupted (0 to skip)')
    parser.add_argument('--python', default=sys.executable, help='Python interpreter for main.py')
    args = parser.parse_args(own)

    specs = (PRESETS[args.preset] if args.preset else []) + args.configs
    if not specs:
        parser.error('give --preset and/or --configs')
    for bad in ('--checkpoint-dir', '--local-checkpoints', '--resume-from', '--model-type', '--seq-model',
                '--visual-pool', '--gate', '--lm-context'):
        if bad in train_args:
            parser.error(f"{bad} is set per configuration by this script; remove it from the pass-through arguments")

    test_subjects = flag_value(train_args, '--test-subjects') or ['s3']
    data_dir = flag_value(train_args, '--data-dir')
    eval_common = (['--test-subjects', *test_subjects] + (['--data-dir', data_dir[0]] if data_dir else [])
                   + [a for f in ('--batch-size', '--num-workers', '--max-samples-per-subject', '--device',
                                  '--frame-size', '--min-detected-ratio')
                      if flag_value(train_args, f) for a in (f, *flag_value(train_args, f))])

    os.makedirs(args.runs_dir, exist_ok=True)
    configs = [parse_config(s) for s in dict.fromkeys(specs)]
    manifest_path = os.path.join(args.runs_dir, 'ablation.json')
    manifest = {'configs': [n for n, _ in configs], 'train_args': train_args, 'corrupt_ratio': args.corrupt_ratio,
                'runs': {}}
    if os.path.exists(manifest_path):
        manifest['runs'] = json.load(open(manifest_path)).get('runs', {})

    print(f"Ablation: {len(configs)} configurations -> {os.path.abspath(args.runs_dir)}")
    for i, (name, cfg) in enumerate(configs, 1):
        run_dir = os.path.join(args.runs_dir, name)
        eval_dir = os.path.join(run_dir, 'evaluation')
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, 'run.log')
        print(f"\n[{i}/{len(configs)}] {name}  ({cfg})", flush=True)
        with open(os.path.join(run_dir, 'run_config.json'), 'w') as f:
            json.dump({**cfg, 'train_args': train_args}, f, indent=2)

        started = time.time()
        status = manifest['runs'].get(name, {})
        if os.path.exists(os.path.join(run_dir, 'test_results.json')):
            print("    training already finished - skipped")
        else:
            cmd = [args.python, 'main.py', 'train', *train_args, '--checkpoint-dir', run_dir,
                   '--model-type', cfg['model_type'], '--seq-model', cfg['seq_model'],
                   '--visual-pool', cfg['visual_pool'], '--gate', cfg['gate'],
                   '--lm-context', cfg['lm_context']]
            last = os.path.join(run_dir, 'last_model.pth')
            if os.path.exists(last):
                cmd += ['--resume-from', last]
                print("    resuming from last_model.pth")
            rc = run(cmd, log_path)
            if rc != 0:
                print(f"    FAILED (exit {rc}) - see {log_path}; continuing with the next configuration")
                status.update({'status': 'failed', 'exit_code': rc})
                manifest['runs'][name] = status
                json.dump(manifest, open(manifest_path, 'w'), indent=2)
                continue
            status['train_seconds'] = status.get('train_seconds', 0) + round(time.time() - started)

        best = os.path.join(run_dir, 'best_model.pth')
        for ratio in [0.0] + ([args.corrupt_ratio] if args.corrupt_ratio > 0 else []):
            suffix = f"_corrupt{ratio:g}" if ratio else ""
            if os.path.exists(os.path.join(eval_dir, f'evaluation_results{suffix}.json')):
                continue
            rc = run([args.python, 'main.py', 'evaluate', *eval_common, '--checkpoint-path', best,
                      '--results-dir', eval_dir] + (['--corrupt-ratio', str(ratio)] if ratio else []), log_path)
            if rc != 0:
                print(f"    evaluation{suffix or ''} FAILED (exit {rc}) - see {log_path}")
        status['status'] = 'done'
        manifest['runs'][name] = status
        json.dump(manifest, open(manifest_path, 'w'), indent=2)

    print(f"\nAblation finished. Compare with: python scripts/compare_runs.py {args.runs_dir}")


if __name__ == '__main__':
    main()
