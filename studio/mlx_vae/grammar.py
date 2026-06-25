"""Grammar-constrained decoding for the Aria AbsTokenizer note/pedal grammar (MLX).

Ported verbatim (logic) from ``src/aria_vae_generate.py``. The AbsTokenizer
detokenizer asserts a strict structure: every NOTE (instrument,pitch,vel) is
followed by ONSET then DUR; every PEDAL by an ONSET; ``<T>`` advances time;
``<E>`` ends. Plain sampling can emit out-of-grammar tokens that make
detokenize raise and drop the whole generation. Masking the logits to the
grammatically valid next-category each step removes those failures by
construction — essential for a real-time stream that must never crash.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

_CAT_NOTE, _CAT_ONSET, _CAT_DUR, _CAT_PEDAL, _CAT_TIME, _CAT_EOS, _CAT_OTHER = range(7)


def build_grammar(tok):
    """Returns dict: cat_list (python, per-id), and mx bool (V,) masks
    top/onset/dur, plus additive (V,) -inf masks neg_top/neg_onset/neg_dur for
    fast logit masking."""
    id_to_tok = {i: t for t, i in tok.tok_to_id.items()}
    V = int(tok.vocab_size)
    instruments = set(tok.instruments_nd)
    time_tok = tok.time_tok
    eos_tok = getattr(tok, "eos_tok", "<E>")
    pedal = set()
    if getattr(tok, "include_pedal", False):
        pedal = {getattr(tok, "ped_on_tok", None), getattr(tok, "ped_off_tok", None)}

    cat = [_CAT_OTHER] * V
    for i in range(V):
        t = id_to_tok.get(i)
        if t == time_tok:
            cat[i] = _CAT_TIME
        elif t in pedal:
            cat[i] = _CAT_PEDAL
        elif t == eos_tok:
            cat[i] = _CAT_EOS
        elif isinstance(t, tuple) and len(t) >= 1 and t[0] == "onset":
            cat[i] = _CAT_ONSET
        elif isinstance(t, tuple) and len(t) >= 1 and t[0] == "dur":
            cat[i] = _CAT_DUR
        elif isinstance(t, tuple) and len(t) >= 1 and t[0] in instruments:
            cat[i] = _CAT_NOTE
    cat_np = np.array(cat)
    top = (cat_np == _CAT_NOTE) | (cat_np == _CAT_PEDAL) | (cat_np == _CAT_TIME) | (cat_np == _CAT_EOS)
    onset = cat_np == _CAT_ONSET
    dur = cat_np == _CAT_DUR

    def neg(mask):
        return mx.array(np.where(mask, 0.0, -np.inf).astype(np.float32))

    nper = {c: int((cat_np == c).sum()) for c in range(7)}
    print(f"[grammar] vocab={V} note={nper[_CAT_NOTE]} onset={nper[_CAT_ONSET]} "
          f"dur={nper[_CAT_DUR]} pedal={nper[_CAT_PEDAL]} time={nper[_CAT_TIME]} "
          f"eos={nper[_CAT_EOS]} other={nper[_CAT_OTHER]}")
    return {
        "cat_list": cat, "V": V,
        "neg_top": neg(top), "neg_onset": neg(onset), "neg_dur": neg(dur),
    }


class GrammarFSM:
    """State machine over the note/pedal grammar. ``neg_mask()`` returns the
    additive (V,) -inf mask for the current state."""

    S_TOP, S_NOTE1, S_NOTE2, S_PED1 = range(4)

    def __init__(self, grammar):
        self.g = grammar
        self.state = self.S_TOP

    def reset(self):
        self.state = self.S_TOP

    def neg_mask(self):
        if self.state == self.S_NOTE2:
            return self.g["neg_dur"]
        if self.state in (self.S_NOTE1, self.S_PED1):
            return self.g["neg_onset"]
        return self.g["neg_top"]

    def advance(self, tid: int):
        c = self.g["cat_list"][tid]
        s = self.state
        if c == _CAT_NOTE:
            self.state = self.S_NOTE1
        elif c == _CAT_PEDAL:
            self.state = self.S_PED1
        elif c == _CAT_ONSET:
            self.state = self.S_NOTE2 if s == self.S_NOTE1 else self.S_TOP
        else:  # DUR, TIME, EOS, OTHER -> event complete
            self.state = self.S_TOP

    def replay(self, ids):
        self.reset()
        for tid in ids:
            self.advance(int(tid))


def sanitize_aria_tokens(toks, tok):
    """Rebuild a strictly grammar-valid token stream before detokenize: keep each
    note only with its onset+dur, each pedal only with its onset; drop stray
    onset/dur; pass prefix/<S>/<T>/<D> through; truncate at <E>."""
    instruments = set(tok.instruments_nd)
    eos_tok = getattr(tok, "eos_tok", "<E>")
    ped = set()
    if getattr(tok, "include_pedal", False):
        ped = {getattr(tok, "ped_on_tok", None), getattr(tok, "ped_off_tok", None)}

    def _is(t, kind):
        return isinstance(t, tuple) and len(t) >= 1 and t[0] == kind

    out, i, n = [], 0, len(toks)
    while i < n:
        t = toks[i]
        if t == eos_tok:
            break
        if isinstance(t, tuple) and len(t) >= 1 and t[0] in instruments:
            if i + 2 < n and _is(toks[i + 1], "onset") and _is(toks[i + 2], "dur"):
                out.extend(toks[i:i + 3]); i += 3
            else:
                i += 1
            continue
        if t in ped:
            if i + 1 < n and _is(toks[i + 1], "onset"):
                out.extend(toks[i:i + 2]); i += 2
            else:
                i += 1
            continue
        if _is(t, "onset") or _is(t, "dur"):
            i += 1
            continue
        out.append(t); i += 1
    return out
