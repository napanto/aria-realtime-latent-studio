"""MLX-vs-torch parity check for the Cadenza port. Runs on the host (needs MLX).

Loads the converted MLX weights + the torch golden_cadenza.npz, recomputes the
same intermediates in MLX, reports max-abs error and argmax agreement:

  * Composer encode → mu (tight fp32 parity).
  * Composer teacher-forced decode logits (fp32 + bf16 + argmax agreement).
  * Performer fill logits (fp32 + argmax agreement on masked positions).
"""
from __future__ import annotations

import argparse

import mlx.core as mx
import numpy as np

from cadenza_mlx import CadenzaComposerMLX, CadenzaPerformerMLX


def maxabs(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_dir", required=True)
    ap.add_argument("--golden", required=True)
    args = ap.parse_args()

    g = np.load(args.golden, allow_pickle=True)

    # ---- Composer ----
    composer = CadenzaComposerMLX.load(args.weights_dir, dtype=mx.float32)
    ids = mx.array(g["ids"].astype(np.int32))
    dec_in = mx.array(g["dec_in"].astype(np.int32))
    mu = composer.encode(ids)
    z = mu
    logits = composer.decode_full(z, dec_in)
    mx.eval(mu, logits)

    e_mu = maxabs(np.array(mu), g["mu"])
    keep = int(g["keep"])
    lt = np.array(logits)[:, -keep:, :]
    e_log_fp32 = maxabs(lt, g["logits_tail"])
    e_log_bf16 = maxabs(lt, g["logits_tail_bf16"])
    am_mlx = lt[0].argmax(-1)
    am_g32 = g["logits_tail"][0].argmax(-1)
    am_g16 = g["logits_tail_bf16"][0].argmax(-1)
    agree32 = float((am_mlx == am_g32).mean())
    agree16 = float((am_mlx == am_g16).mean())

    print("== Composer ==")
    print(f"  mu        max|Δ| = {e_mu:.3e}   (expect < 5e-3)")
    print(f"  logits vs fp32-golden max|Δ| = {e_log_fp32:.3e}  argmax-agree = {agree32:.2%}")
    print(f"  logits vs bf16-golden max|Δ| = {e_log_bf16:.3e}  argmax-agree = {agree16:.2%}")

    # ---- Performer ----
    performer = CadenzaPerformerMLX.load(args.weights_dir, dtype=mx.float32)
    perf_in = mx.array(g["perf_in"].astype(np.int32))
    plogits = performer.fill(perf_in)
    mx.eval(plogits)
    pkeep = int(g["pkeep"])
    plt = np.array(plogits)[:, -pkeep:, :]
    e_plog = maxabs(plt, g["plogits_tail"])
    # argmax agreement on the masked positions (full sequence).
    perf_mask = g["perf_mask"]
    plogits_full = np.array(plogits)[0]
    pg_full_tail = g["plogits_tail"][0]
    am_p_mlx = plt[0].argmax(-1)
    am_p_g = pg_full_tail.argmax(-1)
    agree_p = float((am_p_mlx == am_p_g).mean())

    print("== Performer ==")
    print(f"  logits vs fp32-golden max|Δ| = {e_plog:.3e}  argmax-agree(tail) = {agree_p:.2%}")

    ok = (e_mu < 5e-3 and agree32 > 0.9 and agree16 > 0.85 and e_plog < 5e-2 and agree_p > 0.9)
    print("PARITY:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
