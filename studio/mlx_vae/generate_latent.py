"""Headless AriaVAE-MLX generation with latent control + constrained decoding.

The automatable validation of the whole pipeline (no MIDI hardware needed):
seed MIDI -> encode -> z -> (slider sweep) -> KV-cached constrained generation
-> MIDI, measuring (a) per-token latency (the 'minimal performance penalty'
claim) and (b) that moving an attribute slider actually moves that attribute in
the generated output (latent control works).

Run on the host (mlx + ariautils):
  PYTHONPATH=<repo> python studio/mlx_vae/generate_latent.py \
      --weights_dir weights/mlx --tokenizer demo/demo-tokenizer-config.json \
      --seed_midi example-prompts/pokey_jazz.mid --out_dir /tmp/latent_out \
      --sweep velocity_mean --quantize
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from aria_vae_mlx import AriaVAEMLX
from grammar import GrammarFSM, build_grammar, sanitize_aria_tokens
from latent_control import LatentController


def sample_min_p(logits: mx.array, p_base: float, temp: float) -> int:
    """Min-p sampler in logit space (matches demo_mlx.sample_min_p)."""
    if temp <= 0.0:
        return int(mx.argmax(logits).item())
    logits = logits / temp
    if p_base <= 0.0:
        return int(mx.argmax(logits).item())
    log_p_max = mx.max(logits, axis=-1, keepdims=True)
    thresh = mx.log(mx.array(p_base)) + log_p_max
    masked = mx.where(logits >= thresh, logits, -mx.inf)
    return int(mx.random.categorical(masked).item())


def midi_attrs(md) -> dict:
    """Ground-truth performance attributes from a detokenised MidiDict."""
    notes = md.note_msgs
    if not notes:
        return {}
    vel = np.array([n["data"]["velocity"] for n in notes], float)
    pit = np.array([n["data"]["pitch"] for n in notes], float)
    onsets = np.array([n["data"]["start"] for n in notes], float)
    tpq = md.ticks_per_beat
    # tempo: use first tempo msg if present else 120 bpm
    tempo = 500000
    if getattr(md, "tempo_msgs", None):
        tempo = md.tempo_msgs[0]["data"]
    sec = (onsets.max() - onsets.min()) / tpq * (tempo / 1e6) if onsets.max() > onsets.min() else 1.0
    return {
        "velocity_mean": float(vel.mean()), "velocity_std": float(vel.std()),
        "note_density": float(len(notes) / max(sec, 1e-3)),
        "pitch_mean": float(pit.mean()), "pitch_std": float(pit.std()),
    }


def generate_one(model, tok, grammar, prompt_ids, n_new, temp, min_p, eos_id, constrained):
    """KV-cached constrained generation. Returns (full_ids, tokens_per_sec)."""
    P = len(prompt_ids)
    model.setup_stream(max_seq_len=model.model_config.max_seq_len)
    model.prefill_prefix()
    # prefill the prompt (positions 0..P-1 of real tokens)
    pa = mx.array(np.asarray(prompt_ids, np.int32)[None, :])
    logits = model.forward_cached(pa, mx.arange(P, dtype=mx.int32), 0, P - 1)
    mx.eval(logits)
    last = logits[0, -1]
    fsm = GrammarFSM(grammar) if constrained else None
    if fsm:
        fsm.replay(prompt_ids)
    out = list(prompt_ids)
    t0 = time.time()
    for i in range(n_new):
        lg = last
        if fsm:
            lg = lg + fsm.neg_mask()
        tid = sample_min_p(lg, min_p, temp)
        out.append(tid)
        if fsm:
            fsm.advance(tid)
        if eos_id is not None and tid == eos_id:
            break
        pos = P + i
        logits = model.forward_cached(mx.array([[tid]], dtype=mx.int32),
                                      mx.array([pos], dtype=mx.int32), pos, pos)
        mx.eval(logits)
        last = logits[0, -1]
    dt = time.time() - t0
    n = len(out) - P
    return out, (n / dt if dt > 0 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seed_midi", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--aria_repo", default=None)
    ap.add_argument("--prompt_len", type=int, default=256)
    ap.add_argument("--n_new", type=int, default=384)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--min_p", type=float, default=0.03)
    ap.add_argument("--sweep", default="velocity_mean")
    ap.add_argument("--sweep_cc", default="16,40,64,88,112")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--constrained", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import sys
    if args.aria_repo:
        sys.path.insert(0, args.aria_repo)
    from ariautils.tokenizer import AbsTokenizer
    from ariautils.midi import MidiDict

    mx.random.seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    tok = AbsTokenizer(config_path=args.tokenizer)
    eos_id = tok.tok_to_id.get(getattr(tok, "eos_tok", "<E>"))
    grammar = build_grammar(tok)

    print("[gen] loading AriaVAE-MLX ...")
    model = AriaVAEMLX.load(args.weights_dir, quantize=args.quantize)
    ctrl = LatentController(str(Path(args.weights_dir) / "latent_directions.npz"))
    print("[gen]", ctrl.slider_report())

    # seed -> prompt ids -> base z
    prompt = tok.encode(tok.tokenize(MidiDict.from_midi(args.seed_midi)))[:args.prompt_len]
    mu = model.encode(mx.array(np.asarray(prompt, np.int32)[None, :]))
    mx.eval(mu)
    ctrl.set_base(np.array(mu)[0])
    print(f"[gen] seed '{Path(args.seed_midi).name}' -> {len(prompt)} prompt toks; "
          f"base predicted attrs: { {k: round(v,2) for k,v in ctrl.predicted_attrs().items()} }")

    attr_idx = ctrl.attr_index(args.sweep)
    ccvals = [int(x) for x in args.sweep_cc.split(",")]
    rows, tps_all = [], []
    for cc in ccvals:
        ctrl.clear(); ctrl.set_cc(attr_idx, cc)
        z = ctrl.z()
        model.set_z(mx.array(z))
        full, tps = generate_one(model, tok, grammar, prompt, args.n_new,
                                 args.temp, args.min_p, eos_id, args.constrained)
        tps_all.append(tps)
        toks = sanitize_aria_tokens(tok.decode(full), tok)
        md = tok.detokenize(toks)
        path = out_dir / f"{args.sweep}_cc{cc:03d}.mid"
        md.to_midi().save(str(path))
        a = midi_attrs(md)
        delta = ctrl.cc_to_delta(attr_idx, cc)
        rows.append((cc, delta, a.get(args.sweep, float("nan")), len(full) - len(prompt), tps))
        print(f"[gen] cc={cc:3d} Δ={delta:+6.2f}  measured {args.sweep}={a.get(args.sweep, float('nan')):.2f}  "
              f"gen={len(full)-len(prompt)} toks  {tps:.1f} tok/s -> {path.name}")

    # monotonicity check: measured attribute should track the slider
    meas = [r[2] for r in rows]
    deltas = [r[1] for r in rows]
    corr = float(np.corrcoef(deltas, meas)[0, 1]) if len(set(meas)) > 1 else float("nan")
    print(f"\n[gen] slider->attribute correlation for '{args.sweep}': {corr:+.3f}")
    print(f"[gen] median latency: {np.median(tps_all):.1f} tok/s "
          f"({1000/np.median(tps_all):.1f} ms/token)")
    print("[gen] OK" if corr > 0.5 else "[gen] WEAK CONTROL (corr<=0.5)")


if __name__ == "__main__":
    main()
