#!/usr/bin/env python3
"""CPU smoke test for the latent-manipulation core.

Verifies, against the REAL checkpoints, that:
  1. the AriaVAE / Cadenza state-dict loads,
  2. encode(seed) -> z runs and has the right shape,
  3. a ridge probe fits and yields finite per-attribute directions,
  4. z' = z + alpha * w_attr decodes to a non-empty MIDI continuation.

Runs on CPU so it works inside the Linux mlbox distrobox (no MLX, no MPS).
This is NOT the macOS real-time path — it only exercises the torch VAE core.

Usage:
    ISPR_V2_REPO=/path/to/ispr_v2 \
    LATENT_STUDIO_WEIGHTS=/tmp/ls_weights \
    python scripts/smoke_test.py --model aria_vae --seed-dir /path/to/midis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.registry import get_spec, resolve_asset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=["aria_vae", "cadenza_vae"])
    ap.add_argument("--seed-dir", required=True)
    ap.add_argument("--performer-ckpt", default=None)
    ap.add_argument("--n-windows", type=int, default=24)
    ap.add_argument("--probe-max-steps", type=int, default=48,
                    help="cap Cadenza decode length during probe (CPU speed)")
    args = ap.parse_args()

    import numpy as np

    spec = get_spec(args.model)
    seeds = sorted(str(p) for p in Path(args.seed_dir).glob("*.mid"))[: args.n_windows]
    assert seeds, f"no .mid under {args.seed_dir}"
    print(f"[smoke] {args.model}: {len(seeds)} seed files; checkpoint={spec.primary_weight}")

    if args.model == "aria_vae":
        from latent.aria_vae_backend import AriaVAEBackend

        be = AriaVAEBackend(
            checkpoint=str(spec.primary_weight),
            tokenizer_config=str(resolve_asset(spec.tokenizer_config_local)),
            device_pref="cpu",
        ).load()
    else:
        from latent.cadenza_backend import CadenzaVAEBackend

        be = CadenzaVAEBackend(
            composer_ckpt=str(spec.primary_weight),
            performer_ckpt=args.performer_ckpt,
            device_pref="cpu",
        ).load()
    print("[smoke] (1) state-dict loaded OK")

    z = be.encode(seeds[0])
    assert z.shape == (be.z_dim,), z.shape
    assert np.isfinite(z).all()
    print(f"[smoke] (2) encode -> z shape {z.shape}, |z|={np.linalg.norm(z):.3f}")

    if args.model == "cadenza_vae":
        # CPU decode is slow; keep the probe small + short for a smoke test.
        probe = be.build_probe(
            seeds,
            max_windows=args.n_windows,
            samples_per_seed=2,
            decode_max_steps=args.probe_max_steps,
        )
    else:
        probe = be.build_probe(seeds, max_windows=args.n_windows)
    print("[smoke] (3) ridge probe fit; held-out R²:")
    for a in probe.attr_names:
        d = be.direction(a)
        assert d.shape == (be.z_dim,) and np.isfinite(d).all()
        print(f"        {a:16} R²={probe.r2[a]:+.3f}  |w|={np.linalg.norm(d):.3f}")

    out = Path("/tmp") / f"smoke_{args.model}.mid"
    be.generate_with_offsets(
        z, {probe.attr_names[0]: 1.5}, str(out), temperature=0.95, max_new_tokens=64,
        max_steps=96,
    )
    sz = out.stat().st_size if out.exists() else 0
    print(f"[smoke] (4) decode(z + 1.5*w[{probe.attr_names[0]}]) -> {out} ({sz} bytes)")
    assert sz > 0, "decode produced no MIDI"
    print("[smoke] ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
