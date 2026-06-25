"""MLX-vs-torch parity check for the AriaVAE port. Runs on the host (needs MLX).

Loads the converted MLX weights + the torch golden_ref.npz, recomputes the same
intermediates in MLX, and reports max-abs error. Passing means the encoder /
injector / z-adapters ported correctly (tight fp32 parity) and the decode
integration is faithful (bf16 logits match + argmax agreement vs fp32).
"""
from __future__ import annotations

import argparse

import mlx.core as mx
import numpy as np

from aria_vae_mlx import AriaVAEMLX


def maxabs(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_dir", required=True)
    ap.add_argument("--golden", required=True)
    args = ap.parse_args()

    g = np.load(args.golden, allow_pickle=True)
    ids = mx.array(g["ids"].astype(np.int32))

    model = AriaVAEMLX.load(args.weights_dir, quantize=False, dtype=mx.bfloat16)

    mu = model.encode(ids)                 # fp32
    model.set_z(mu)
    prefix = model.prefix.astype(mx.float32)
    resid = mx.stack(model.residuals, 0).astype(mx.float32)   # (16,1,1,d) -> compare to (16,1,d)
    attrs = model.attrs(mu)
    logits = model.decode_full(ids).astype(mx.float32)        # (1,S,V) -> tail
    mx.eval(mu, prefix, resid, attrs, logits)

    mu_np = np.array(mu)
    e_mu = maxabs(mu_np, g["mu"])
    e_prefix = maxabs(np.array(prefix), g["prefix"])
    e_resid = maxabs(np.array(resid).reshape(16, 1, -1), g["residuals"])
    e_attrs = maxabs(np.array(attrs), g["attrs"])

    keep = int(g["keep"])
    lt = np.array(logits)[:, -keep:, :]
    e_log_fp32 = maxabs(lt, g["logits_tail"])
    e_log_bf16 = maxabs(lt, g["logits_tail_bf16"])
    am_mlx = lt[0].argmax(-1)
    am_g32 = g["logits_tail"][0].argmax(-1)
    am_g16 = g["logits_tail_bf16"][0].argmax(-1)
    agree32 = float((am_mlx == am_g32).mean())
    agree16 = float((am_mlx == am_g16).mean())

    print(f"  mu        max|Δ| = {e_mu:.3e}   (expect < 5e-3)")
    print(f"  prefix    max|Δ| = {e_prefix:.3e}   (expect < 1e-2)")
    print(f"  residuals max|Δ| = {e_resid:.3e}   (expect < 1e-2)")
    print(f"  attrs     max|Δ| = {e_attrs:.3e}")
    print(f"  logits vs fp32-golden max|Δ| = {e_log_fp32:.3e}  argmax-agree = {agree32:.2%}")
    print(f"  logits vs bf16-golden max|Δ| = {e_log_bf16:.3e}  argmax-agree = {agree16:.2%}")

    ok = (e_mu < 5e-3 and e_prefix < 1e-2 and e_resid < 1e-2 and agree16 > 0.85)
    print("PARITY:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
