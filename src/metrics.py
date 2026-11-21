"""Evaluation metrics and the shared evaluation loop (used by the trainer and by main.py evaluate).

  FrameMeter     per-frame viseme / phoneme accuracy (all, speech-only, by ground-truth ambiguity),
                 phoneme accuracy within each viseme class, confusion matrices
  SentenceMeter  word / character error rate of decoded sentences, sentence accuracy, per-slot
                 accuracy, letter-slot homophene analysis
  GateMeter      average language-model weight of the fusion gate by frame type
                 (intrinsically ambiguous or not x extrinsically unreliable or not)
"""
import contextlib
from collections import defaultdict

import numpy as np
import torch

from src.config import Config
from src.decoding import (GRID_SLOTS, SLOT_NAMES, char_errors, constrained_decode, greedy_decode,
                          word_errors)
from src.viseme_mapping import viseme_mapper

SILENCE_VISEME = viseme_mapper.get_viseme('sil')
REQUESTED_LETTER_PAIRS = [('b', 'p'), ('d', 't'), ('m', 'n')]


def letter_homophene_groups():
    """GRID letters whose spoken phonemes map to the same viseme sequence (look identical on the lips)."""
    from src.data_loader import GRID_WORD_TO_PHONEMES
    groups = defaultdict(list)
    for letter in Config.LETTERS:
        groups[tuple(viseme_mapper.phoneme_to_viseme_sequence(GRID_WORD_TO_PHONEMES[letter]))].append(letter)
    return [sorted(g) for g in groups.values() if len(g) > 1]


HOMOPHENE_GROUPS = letter_homophene_groups()
LETTER_TO_GROUP = {letter: tuple(g) for g in HOMOPHENE_GROUPS for letter in g}


def _ratio(num, den):
    return float(num) / den if den else 0.0


class FrameMeter:
    def __init__(self, num_visemes=12, num_phonemes=len(Config.PHONEMES)):
        self.c = defaultdict(int)
        self.viseme_cm = np.zeros((num_visemes, num_visemes), dtype=np.int64)
        self.phoneme_cm = np.zeros((num_phonemes, num_phonemes), dtype=np.int64)

    def update(self, viseme_logits, phoneme_logits, visemes, phonemes, ambiguity):
        vp = viseme_logits.argmax(-1).cpu()
        pp = phoneme_logits.argmax(-1).cpu()
        v, p, a = visemes.cpu(), phonemes.cpu(), ambiguity.cpu()
        valid = (v >= 0) & (p >= 0)
        speech = valid & (v != SILENCE_VISEME)
        v_ok, p_ok = vp == v, pp == p
        c = self.c
        for name, mask in (('all', valid), ('speech', speech),
                           ('high', valid & (a > Config.AMBIGUITY_THRESHOLD)),
                           ('low', valid & (a <= Config.AMBIGUITY_THRESHOLD))):
            c[f'{name}_n'] += int(mask.sum())
            c[f'{name}_v'] += int(v_ok[mask].sum())
            c[f'{name}_p'] += int(p_ok[mask].sum())
        np.add.at(self.viseme_cm, (v[valid].numpy(), vp[valid].numpy()), 1)
        np.add.at(self.phoneme_cm, (p[valid].numpy(), pp[valid].numpy()), 1)

    def results(self):
        c = self.c
        # phoneme accuracy within each viseme class (how well homophenous phonemes are told apart)
        within = {}
        for vid, members in sorted(viseme_mapper.viseme_to_phonemes.items()):
            idx = [Config.PHONEMES.index(ph) for ph in members if ph in Config.PHONEMES]
            total = self.phoneme_cm[idx].sum()
            if total:
                within[f'V{vid}'] = {'phonemes': [Config.PHONEMES[i] for i in idx], 'frames': int(total),
                                     'phoneme_accuracy': _ratio(self.phoneme_cm[idx, idx].sum(), total)}
        return {
            'viseme_accuracy': _ratio(c['all_v'], c['all_n']),
            'phoneme_accuracy': _ratio(c['all_p'], c['all_n']),
            'speech_viseme_accuracy': _ratio(c['speech_v'], c['speech_n']),
            'speech_phoneme_accuracy': _ratio(c['speech_p'], c['speech_n']),
            'high_ambiguity_phoneme_accuracy': _ratio(c['high_p'], c['high_n']),
            'low_ambiguity_phoneme_accuracy': _ratio(c['low_p'], c['low_n']),
            'total_frames': c['all_n'], 'speech_frames': c['speech_n'],
            'phoneme_accuracy_within_viseme_class': within,
        }


class SentenceMeter:
    """Error rates of one decoding method against the reference sentences."""

    def __init__(self):
        self.word_err = self.word_n = self.char_err = self.char_n = 0
        self.sentences = self.exact = self.well_formed = 0
        self.slot_ok = [0] * len(GRID_SLOTS)
        self.letter = defaultdict(int)
        self.pair_confusions = {f'{a}/{b}': 0 for a, b in REQUESTED_LETTER_PAIRS}
        self.pair_totals = {f'{a}/{b}': 0 for a, b in REQUESTED_LETTER_PAIRS}

    def update(self, refs, hyps):
        for ref, hyp in zip(refs, hyps):
            we, wn = word_errors(ref, hyp)
            ce, cn = char_errors(ref, hyp)
            self.word_err += we; self.word_n += wn; self.char_err += ce; self.char_n += cn
            self.sentences += 1
            self.exact += ref == hyp
            r, h = ref.split(), hyp.split()
            if len(r) != len(GRID_SLOTS):
                continue
            if len(h) == len(GRID_SLOTS):
                self.well_formed += 1
                for i in range(len(GRID_SLOTS)):
                    self.slot_ok[i] += r[i] == h[i]
            # letter slot (slot 3); a malformed hypothesis counts as an error
            true_letter = r[3]
            pred_letter = h[3] if len(h) == len(GRID_SLOTS) else None
            group = LETTER_TO_GROUP.get(true_letter)
            kind = 'homophene' if group else 'distinct'
            self.letter[f'{kind}_n'] += 1
            self.letter[f'{kind}_ok'] += pred_letter == true_letter
            if group and pred_letter != true_letter and pred_letter in group:
                self.letter['within_group_confusions'] += 1
            for a, b in REQUESTED_LETTER_PAIRS:
                if true_letter in (a, b):
                    key = f'{a}/{b}'
                    self.pair_totals[key] += 1
                    self.pair_confusions[key] += pred_letter == (b if true_letter == a else a)

    def results(self):
        n = self.sentences
        return {
            'wer': _ratio(self.word_err, self.word_n),
            'cer': _ratio(self.char_err, self.char_n),
            'sentence_accuracy': _ratio(self.exact, n),
            'well_formed_rate': _ratio(self.well_formed, n),
            'slot_accuracy': {name: _ratio(ok, self.well_formed) for name, ok in zip(SLOT_NAMES, self.slot_ok)},
            'letter_slot': {
                'accuracy': _ratio(self.letter['homophene_ok'] + self.letter['distinct_ok'],
                                   self.letter['homophene_n'] + self.letter['distinct_n']),
                'homophene_letter_accuracy': _ratio(self.letter['homophene_ok'], self.letter['homophene_n']),
                'distinct_letter_accuracy': _ratio(self.letter['distinct_ok'], self.letter['distinct_n']),
                'homophene_letters': self.letter['homophene_n'],
                'within_group_confusion_rate': _ratio(self.letter['within_group_confusions'], self.letter['homophene_n']),
                'pair_confusion_rate': {k: _ratio(self.pair_confusions[k], self.pair_totals[k]) for k in self.pair_totals},
                'pair_counts': dict(self.pair_totals),
            },
            'sentences': n,
        }


class GateMeter:
    """Mean language-model weight of the fusion gate by frame type."""

    CATEGORIES = ('clear_reliable', 'ambiguous_reliable', 'clear_unreliable', 'ambiguous_unreliable')

    def __init__(self):
        self.sum = defaultdict(float)
        self.n = defaultdict(int)

    def update(self, predictions, visemes, gt_ambiguity, quality):
        fw = predictions.get('fusion_weights')
        if fw is None:
            return
        w_lang = torch.stack([f[..., 1] for f in fw]).mean(0).float().cpu()
        valid = (visemes >= 0).cpu()
        ambiguous = (gt_ambiguity.cpu() > Config.AMBIGUITY_THRESHOLD)
        unreliable = (quality.cpu()[..., 0] > 0)
        extra = {k: predictions.get(k) for k in ('intrinsic_logit', 'extrinsic_logit')}
        for name, mask in zip(self.CATEGORIES, (~ambiguous & ~unreliable, ambiguous & ~unreliable,
                                                ~ambiguous & unreliable, ambiguous & unreliable)):
            mask = mask & valid
            self.n[name] += int(mask.sum())
            self.sum[name] += float(w_lang[mask].sum())
            for k, v in extra.items():
                if v is not None:
                    self.sum[f'{name}_{k}'] += float(v.float().cpu()[mask].sum())

    def results(self):
        if not self.n:
            return None
        out = {}
        for name in self.CATEGORIES:
            n = self.n[name]
            entry = {'frames': n, 'mean_language_weight': _ratio(self.sum[name], n)}
            for k in ('intrinsic_logit', 'extrinsic_logit'):
                if f'{name}_{k}' in self.sum:
                    entry[f'mean_{k}'] = _ratio(self.sum[f'{name}_{k}'], n)
            out[name] = entry
        return out


def batch_to_device(batch, device):
    return {
        'frames': batch['frames'].to(device, non_blocking=True),
        'lengths': batch['lengths'].to(device),
        'quality': batch['quality'].to(device),
        'targets': {'targets': batch['targets'], 'target_lengths': batch['target_lengths'],
                    'visemes': batch['visemes'].to(device), 'phonemes': batch['phonemes'].to(device),
                    'words': batch['word_labels'].to(device), 'detected': batch['detected'].to(device)},
    }


def autocast_context(device, enabled):
    if enabled and str(device).startswith('cuda'):
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    return contextlib.nullcontext()


def run_evaluation(model, loader, device, full=False, amp=False, max_examples=20, progress=None):
    """Evaluate a model on a loader.

    full=False: losses, frame metrics and greedy-decoding error rates (cheap; used every epoch).
    full=True: additionally grammar-constrained decoding, gate analysis, frame-level confusion
    matrices and example predictions.
    """
    model.eval()
    loss_sums, loss_n = defaultdict(float), 0
    frames, greedy, constrained, gates = FrameMeter(), SentenceMeter(), SentenceMeter(), GateMeter()
    examples = []
    iterator = progress(loader) if progress else loader
    with torch.no_grad():
        for batch in iterator:
            d = batch_to_device(batch, device)
            speaker_ids = model.speaker_ids_for(batch['subjects'], device)
            with autocast_context(device, amp):
                p = model(d['frames'], lengths=d['lengths'], speaker_ids=speaker_ids, quality=d['quality'])
                losses = model.compute_loss(p, d['targets'])
            for k, v in losses.items():
                loss_sums[k] += float(v)
            loss_n += 1

            frames.update(p['viseme_logits'], p['phoneme_logits'], d['targets']['visemes'],
                          d['targets']['phonemes'], batch['ambiguity_scores'])
            lp = p['ctc_log_probs'].float().cpu().numpy()
            lens = batch['lengths'].tolist()
            hyp_g = [greedy_decode(lp[i], lens[i]) for i in range(len(lens))]
            greedy.update(batch['sentences'], hyp_g)
            if full:
                hyp_c = [constrained_decode(lp[i], lens[i]) for i in range(len(lens))]
                constrained.update(batch['sentences'], hyp_c)
                gates.update(p, d['targets']['visemes'], batch['ambiguity_scores'], d['quality'])
                for i in range(len(lens)):
                    if len(examples) < max_examples:
                        examples.append({'subject': batch['subjects'][i], 'utterance': batch['utterance_ids'][i],
                                         'reference': batch['sentences'][i], 'greedy': hyp_g[i],
                                         'constrained': hyp_c[i]})

    result = {'losses': {k: v / max(loss_n, 1) for k, v in loss_sums.items()},
              'frame': frames.results(), 'greedy': greedy.results()}
    if full:
        result.update({'constrained': constrained.results(), 'gate': gates.results(), 'examples': examples,
                       'homophene_groups': HOMOPHENE_GROUPS})
        result['_viseme_cm'] = frames.viseme_cm
        result['_phoneme_cm'] = frames.phoneme_cm
    return result
