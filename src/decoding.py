"""Sentence-level CTC utilities: character encoding, greedy and GRID-grammar-constrained decoding,
and word / character error rates.

Output alphabet (Config.CTC_CHARS): index 0 is the CTC blank, then space and a-z.
"""
import math

import numpy as np

from src.config import Config

BLANK = Config.CTC_BLANK
# index -> character (index 0 is the blank, shown as '')
INDEX_TO_CHAR = [''] + list(Config.CTC_CHARS)
CHAR_TO_INDEX = {c: i for i, c in enumerate(INDEX_TO_CHAR) if c}
NUM_CLASSES = len(INDEX_TO_CHAR)

# The fixed GRID sentence grammar: command colour preposition letter digit adverb
GRID_SLOTS = [Config.COMMANDS, Config.COLORS, Config.PREPOSITIONS, Config.LETTERS, Config.DIGITS, Config.ADVERBS]
SLOT_NAMES = ['command', 'color', 'preposition', 'letter', 'digit', 'adverb']


# ------------------------------------------------------------------ encoding
def sentence_from_words(words):
    """GRID word list (may contain 'sil'/'sp') -> lowercase sentence string of the spoken words."""
    return ' '.join(w for w in words if w not in ('sil', 'sp'))


def text_to_indices(text):
    return [CHAR_TO_INDEX[c] for c in text]


def indices_to_text(indices):
    return ''.join(INDEX_TO_CHAR[i] for i in indices if i != BLANK)


# ------------------------------------------------------------------ greedy
def greedy_decode(log_probs, length=None):
    """Best-path decoding: argmax per frame, merge repeats, drop blanks.

    log_probs: (T, C) array/tensor of per-frame log-probabilities.
    """
    lp = np.asarray(log_probs)
    if length is not None:
        lp = lp[:length]
    best = lp.argmax(axis=-1)
    out, prev = [], None
    for idx in best:
        if idx != prev and idx != BLANK:
            out.append(int(idx))
        prev = idx
    return ' '.join(indices_to_text(out).split())  # normalise stray/double spaces


# ------------------------------------------------------------------ grammar-constrained beam search
class _GridGrammar:
    """Which characters may follow a partial sentence under the GRID grammar."""

    def __init__(self):
        self.prefixes = []   # per slot: every prefix of every word
        self.words = []      # per slot: complete words
        for vocab in GRID_SLOTS:
            self.words.append(set(vocab))
            self.prefixes.append({w[:i] for w in vocab for i in range(1, len(w) + 1)})

    def allowed(self, text):
        """Set of characters that keep `text` a prefix of some grammatical sentence."""
        parts = text.split(' ')
        slot, partial = len(parts) - 1, parts[-1]
        if slot >= len(GRID_SLOTS):
            return set()
        allowed = {c for c in Config.CTC_CHARS[1:] if partial + c in self.prefixes[slot]}
        if partial in self.words[slot] and slot < len(GRID_SLOTS) - 1:
            allowed.add(' ')
        return allowed

    def complete(self, text):
        parts = text.split(' ')
        return len(parts) == len(GRID_SLOTS) and all(p in w for p, w in zip(parts, self.words))


_GRAMMAR = _GridGrammar()


def _logsumexp(*xs):
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in xs))


def constrained_decode(log_probs, length=None, beam_size=None):
    """CTC prefix beam search restricted to grammatical GRID sentences.

    Every hypothesis is kept a prefix of some valid "command colour preposition letter digit
    adverb" sentence, so the result is always six in-vocabulary words when any complete
    hypothesis survives (otherwise the best partial hypothesis is returned).
    """
    beam_size = beam_size or Config.BEAM_SIZE
    lp = np.asarray(log_probs, dtype=np.float64)
    if length is not None:
        lp = lp[:length]
    neg_inf = -math.inf
    # prefix -> (log p ending in blank, log p ending in non-blank)
    beams = {'': (0.0, neg_inf)}
    for t in range(lp.shape[0]):
        frame = lp[t]
        next_beams = {}

        def add(prefix, pb, pnb):
            ob, onb = next_beams.get(prefix, (neg_inf, neg_inf))
            next_beams[prefix] = (_logsumexp(ob, pb), _logsumexp(onb, pnb))

        for prefix, (pb, pnb) in beams.items():
            total = _logsumexp(pb, pnb)
            # stay: blank, or repeat of the last character without a blank in between
            add(prefix, total + frame[BLANK], neg_inf)
            if prefix:
                add(prefix, neg_inf, pnb + frame[CHAR_TO_INDEX[prefix[-1]]])
            for c in _GRAMMAR.allowed(prefix):
                ci = CHAR_TO_INDEX[c]
                if prefix and c == prefix[-1]:
                    add(prefix + c, neg_inf, pb + frame[ci])     # repeat needs a blank between
                else:
                    add(prefix + c, neg_inf, total + frame[ci])
        beams = dict(sorted(next_beams.items(), key=lambda kv: _logsumexp(*kv[1]), reverse=True)[:beam_size])

    ranked = sorted(beams.items(), key=lambda kv: _logsumexp(*kv[1]), reverse=True)
    for prefix, _ in ranked:
        if _GRAMMAR.complete(prefix):
            return prefix
    return ranked[0][0] if ranked else ''


# ------------------------------------------------------------------ error rates
def edit_distance(ref, hyp):
    """Levenshtein distance between two sequences."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def word_errors(ref, hyp):
    """(edit distance in words, number of reference words)"""
    r = ref.split()
    return edit_distance(r, hyp.split()), len(r)


def char_errors(ref, hyp):
    """(edit distance in characters, number of reference characters)"""
    return edit_distance(list(ref), list(hyp)), len(ref)
