"""Headless Cadenza-MLX latent-control validation (host: mlx + miditok).

Mirrors ``generate_latent.py`` (AriaVAE), adapted to the Cadenza two-stage VAE:

  seed MIDI -> encode -> base z (Composer μ) -> for each slider value:
      z' = base + Δ·(w_k/||w_k||²)  -> Composer AR-generate -> (mask + Performer
      fill) -> detok MIDI -> measure the swept attribute on the output.

Reports the slider->attribute correlation (does moving the latent slider move
the generated attribute?) plus per-phase latency. Uses the fitted
``latent_directions_cadenza.npz`` for the calibrated control axes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from cadenza_mlx import CadenzaComposerMLX, CadenzaPerformerMLX
from latent_control import LatentController


PERFORMANCE_PREFIXES = ("Velocity_", "MicroTiming_", "Pedal_", "PedalOff_")


def load_perf_tok(ispr_path: str):
    sys.path.insert(0, ispr_path)
    from src.data.pertok_tokenizer import PerTokWrapper
    return PerTokWrapper.from_default(cache_root=None, mode="performance")


def measured_attr(perf_tok, ids: np.ndarray, name: str):
    """Decode ids -> MIDI -> the named attribute (eval_cadenza_stagec set)."""
    # local attribute computation mirroring src.eval_cadenza_stagec
    try:
        pm = perf_tok.decode(ids)
    except Exception:
        return float("nan")
    pitches, starts, vels = [], [], []
    for inst in pm.instruments:
        for n in inst.notes:
            pitches.append(n.pitch); starts.append(n.start); vels.append(n.velocity)
    if not pitches:
        return float("nan")
    p = np.asarray(pitches, float); v = np.asarray(vels, float)
    s = np.sort(np.asarray(starts, float))
    dur = max(pm.get_end_time(), 1e-3)
    if name == "velocity_mean": return float(v.mean())
    if name == "velocity_std": return float(v.std())
    if name == "note_density": return float(len(p) / dur)
    if name == "pitch_mean": return float(p.mean())
    if name == "pitch_std": return float(p.std())
    if name == "ioi_entropy":
        iois = np.diff(s)
        if iois.size == 0: return 0.0
        hist, _ = np.histogram(iois, bins=np.linspace(0, 1.2, 13))
        h = hist.astype(float); tot = h.sum()
        if tot == 0: return 0.0
        pr = h / tot; nz = pr[pr > 0]
        return float(-np.sum(nz * np.log(nz)))
    return float("nan")


def two_stage_ids(composer, performer, perf_tok, perf_id_set, z, *, max_steps,
                  temperature, top_k, top_p, key, performer_sample):
    """z -> composer AR gen -> mask perf slots -> performer fill -> final ids.

    Returns (composer_only_ids, two_stage_ids). The Composer's raw emission
    carries the velocity/microtime the latent directly controls; the Performer
    re-fills those slots (so its output reflects the Performer's prediction for
    velocity/microtime but preserves the Composer's pitch/timeshift/duration).
    """
    pad_id = int(perf_tok.pad_id); eos_id = int(perf_tok.eos_id); mask_id = int(perf_tok.mask_id)
    gen = composer.generate(z, max_steps=max_steps, temperature=temperature,
                            top_k=top_k, top_p=top_p, key=key)
    mx.eval(gen)
    g = np.array(gen[0]).astype(np.int32)
    eos_pos = np.where(g == eos_id)[0]
    if eos_pos.size: g = g[: int(eos_pos[0])]
    nonpad = np.where(g != pad_id)[0]
    if nonpad.size: g = g[: int(nonpad[-1]) + 1]
    if g.size < 4:
        return g, g
    perf_in = g.copy()
    mpos = np.array([t in perf_id_set for t in g], dtype=bool)
    perf_in[mpos] = mask_id
    plen = min(perf_in.size, performer.max_seq_len)
    perf_in = perf_in[:plen]; mpos = mpos[:plen]
    plog = performer.fill(mx.array(perf_in[None, :])); mx.eval(plog)
    plog = np.array(plog[0])
    out = perf_in.copy()
    midx = np.where(mpos)[0]
    if performer_sample:
        for q in midx:
            row = plog[q] - plog[q].max()
            pr = np.exp(row); pr /= pr.sum()
            out[q] = int(np.random.choice(len(pr), p=pr))
    else:
        out[midx] = plog[midx].argmax(-1).astype(np.int32)
    return g, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_dir", required=True)
    ap.add_argument("--seed_midi", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ispr", default=None)
    ap.add_argument("--prompt_len", type=int, default=384)
    ap.add_argument("--max_generate_steps", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=24)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--sweep", default="velocity_mean")
    ap.add_argument("--sweep_cc", default="16,40,64,88,112")
    ap.add_argument("--performer_sample", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mx.random.seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    perf_tok = load_perf_tok(args.ispr) if args.ispr else load_perf_tok(
        str(Path(__file__).resolve().parents[2]))
    perf_id_set = set(perf_tok.token_ids_by_prefix(PERFORMANCE_PREFIXES))

    composer = CadenzaComposerMLX.load(args.weights_dir, dtype=mx.float32)
    performer = CadenzaPerformerMLX.load(args.weights_dir, dtype=mx.float32)
    ctrl = LatentController(str(Path(args.weights_dir) / "latent_directions_cadenza.npz"))
    print("[gen]", ctrl.slider_report())

    prompt = np.asarray(perf_tok.encode_midi(args.seed_midi), np.int32)[:args.prompt_len]
    mu = composer.encode(mx.array(prompt[None, :])); mx.eval(mu)
    ctrl.set_base(np.array(mu)[0])
    print(f"[gen] seed '{Path(args.seed_midi).name}' -> {prompt.size} toks; base attrs: "
          f"{ {k: round(v,2) for k,v in ctrl.predicted_attrs().items()} }")

    attr_idx = ctrl.attr_index(args.sweep)
    ccvals = [int(x) for x in args.sweep_cc.split(",")]
    rows = []; lat = []
    for cc in ccvals:
        ctrl.clear(); ctrl.set_cc(attr_idx, cc)
        z = mx.array(ctrl.z()[None, :])
        t0 = time.time()
        comp_ids, ts_ids = two_stage_ids(composer, performer, perf_tok, perf_id_set, z,
                                         max_steps=args.max_generate_steps,
                                         temperature=args.temperature, top_k=args.top_k,
                                         top_p=args.top_p, key=mx.random.key(args.seed * 100 + cc),
                                         performer_sample=args.performer_sample)
        dt = time.time() - t0; lat.append(dt)
        path = out_dir / f"{args.sweep}_cc{cc:03d}.mid"
        try:
            perf_tok.decode(ts_ids).write(str(path))
        except Exception as e:
            print(f"[gen] cc={cc} write failed: {e}")
        # Velocity/microtime are re-filled by the Performer, so for those
        # attributes measure the Composer's RAW emission (what the latent
        # directly controls). Pitch/density/IOI survive the Performer pass, so
        # measure those on the two-stage output.
        perf_attrs = {"velocity_mean", "velocity_std"}
        meas_ids = comp_ids if args.sweep in perf_attrs else ts_ids
        meas = measured_attr(perf_tok, meas_ids, args.sweep)
        delta = ctrl.cc_to_delta(attr_idx, cc)
        rows.append((cc, delta, meas))
        print(f"[gen] cc={cc:3d} Δ={delta:+7.2f}  measured {args.sweep}={meas:.2f}  "
              f"({ts_ids.size} toks, {dt*1000:.0f} ms) -> {path.name}")

    meas = [r[2] for r in rows if not np.isnan(r[2])]
    deltas = [r[1] for r in rows if not np.isnan(r[2])]
    corr = float(np.corrcoef(deltas, meas)[0, 1]) if len(set(meas)) > 1 else float("nan")
    print(f"\n[gen] slider->attribute correlation for '{args.sweep}': {corr:+.3f}")
    print(f"[gen] median latency: {np.median(lat)*1000:.0f} ms/sample")
    print("[gen] OK" if (not np.isnan(corr) and corr > 0.5) else "[gen] WEAK CONTROL (corr<=0.5 or nan)")
    (out_dir / f"sweep_{args.sweep}.json").write_text(json.dumps(
        {"sweep": args.sweep, "rows": rows, "corr": corr,
         "median_latency_ms": float(np.median(lat) * 1000)}, indent=2))


if __name__ == "__main__":
    main()
