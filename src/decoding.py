"""Sentence-level CTC utilities: character encoding, greedy and GRID-grammar-constrained decoding,
and word / character error rates.

Output alphabet (Config.CTC_CHARS): index 0 is the CTC blank, then space and a-z.
"""
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


class _GridCTCGraph:
    """CTC topology of the GRID grammar, for best-path (Viterbi) decoding.

    Each slot word is spelled as its characters (words after the first are prefixed with a space);
    every character has an emitting state and a following blank state, plus one leading blank.
    Transitions follow CTC rules: stay, move to the next character (directly unless it repeats the
    previous one), or pass through a blank; the first character of a word in slot k is reachable from
    the end (last character or its blank) of any word in slot k-1.
    """

    def __init__(self):
        labels, slot_word, preds = [BLANK], [(-1, None)], [[0]]   # state 0 = leading blank
        ends_prev = None
        start_states = [0]
        for slot, vocab in enumerate(GRID_SLOTS):
            ends = []
            for word in vocab:
                token = word if slot == 0 else ' ' + word
                prev_emit = prev_blank = None
                for i, ch in enumerate(token):
                    emit = len(labels)
                    labels.append(CHAR_TO_INDEX[ch]); slot_word.append((slot, word))
                    if i == 0:
                        p = [emit] + ([0] if slot == 0 else ends_prev)
                        if slot == 0:
                            start_states.append(emit)
                    else:
                        p = [emit, prev_blank] + ([prev_emit] if ch != token[i - 1] else [])
                    preds.append(p)
                    blank = len(labels)
                    labels.append(BLANK); slot_word.append((slot, word)); preds.append([blank, emit])
                    prev_emit, prev_blank = emit, blank
                ends += [prev_emit, prev_blank]
            ends_prev = ends
        self.labels = np.array(labels)
        self.slot_word = slot_word
        self.final_states = np.array(ends_prev)
        self.start_states = np.array(start_states)
        width = max(len(p) for p in preds)
        self.pred = np.array([p + [p[0]] * (width - len(p)) for p in preds])  # pad with a duplicate


_GRAPH = None


def constrained_decode(log_probs, length=None, beam_size=None):
    """Best CTC path restricted to grammatical GRID sentences (Viterbi over the grammar graph).

    Always returns a complete "command colour preposition letter digit adverb" sentence when the clip
    has enough frames; otherwise falls back to greedy decoding. beam_size is accepted for backward
    compatibility and ignored (the search is exact).
    """
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _GridCTCGraph()
    g = _GRAPH
    lp = np.asarray(log_probs, dtype=np.float64)
    if length is not None:
        lp = lp[:length]
    T = lp.shape[0]
    if T == 0:
        return ''
    emit = lp[:, g.labels]                                   # (T, S)
    delta = np.full(len(g.labels), -np.inf)
    delta[g.start_states] = emit[0, g.start_states]
    back = np.zeros((T, len(g.labels)), dtype=np.int32)
    back[0] = np.arange(len(g.labels))
    for t in range(1, T):
        cand = delta[g.pred]                                 # (S, P)
        arg = cand.argmax(axis=1)
        back[t] = g.pred[np.arange(len(g.labels)), arg]
        delta = cand[np.arange(len(g.labels)), arg] + emit[t]
    final = g.final_states[np.argmax(delta[g.final_states])]
    if not np.isfinite(delta[final]):
        return greedy_decode(lp)
    words, state = [None] * len(GRID_SLOTS), final
    for t in range(T - 1, -1, -1):
        slot, word = g.slot_word[state]
        if slot >= 0:
            words[slot] = word
        state = back[t, state]
    return ' '.join(w for w in words if w)


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
