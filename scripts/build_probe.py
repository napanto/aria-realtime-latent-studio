#!/usr/bin/env python3
"""Build the ridge probe (z -> attribute directions) for a VAE backend.

The per-attribute slider directions are the columns of a closed-form ridge
probe fit on encoded seed windows. Upstream persists only R², not the weights,
so we recompute and cache them to ``weights/<model>/probe.npz``.

Usage (inside the Apple-Silicon / mlbox env with torch available):
    ISPR_V2_REPO=/path/to/ispr_v2 \
    python scripts/build_probe.py --model aria_vae   --seed-dir assets/seed_midi
    python scripts/build_probe.py --model cadenza_vae --seed-dir assets/seed_midi \
        --performer-ckpt /path/to/performer.pt
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
    ap.add_argument("--seed-dir", required=True, help="dir of seed .mid files")
    ap.add_argument("--performer-ckpt", default=None)
    ap.add_argument("--max-windows", type=int, default=300)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    spec = get_spec(args.model)
    seeds = sorted(str(p) for p in Path(args.seed_dir).glob("*.mid"))
    if not seeds:
        print(f"no .mid files under {args.seed_dir}", file=sys.stderr)
        return 2
    print(f"{len(seeds)} seed files")

    out_probe = spec.weights_subdir / "probe.npz"

    if args.model == "aria_vae":
        from latent.aria_vae_backend import AriaVAEBackend

        be = AriaVAEBackend(
            checkpoint=str(spec.primary_weight),
            tokenizer_config=str(resolve_asset(spec.tokenizer_config_local)),
            device_pref=args.device,
        ).load()
        probe = be.build_probe(
            seeds, max_windows=args.max_windows, save_to=str(out_probe)
        )
    else:
        from latent.cadenza_backend import CadenzaVAEBackend

        be = CadenzaVAEBackend(
            composer_ckpt=str(spec.primary_weight),
            performer_ckpt=args.performer_ckpt,
            device_pref=args.device,
        ).load()
        probe = be.build_probe(
            seeds, max_windows=args.max_windows, save_to=str(out_probe)
        )

    print(f"saved probe -> {out_probe}")
    print("per-attribute held-out R²:")
    for a in probe.attr_names:
        print(f"  {a:16} {probe.r2[a]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
