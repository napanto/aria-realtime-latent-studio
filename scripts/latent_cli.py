#!/usr/bin/env python3
"""Headless latent generation — drive a VAE backend without the GUI.

Useful for smoke tests and batch demos:

    ISPR_V2_REPO=/path/to/ispr_v2 python scripts/latent_cli.py \
        --model aria_vae --seed assets/seed_midi/foo.mid \
        --set velocity_mean=+1.5 --set note_density=-1.0 \
        --out out.mid
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.registry import get_spec, resolve_asset  # noqa: E402


def _parse_set(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k.strip()] = float(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=["aria_vae", "cadenza_vae"])
    ap.add_argument("--seed", default=None, help="seed .mid (omit for random z)")
    ap.add_argument("--set", action="append", help="attr=alpha (repeatable)")
    ap.add_argument("--performer-ckpt", default=None)
    ap.add_argument("--probe", default=None, help="probe.npz (default weights/<m>/probe.npz)")
    ap.add_argument("--out", default="out.mid")
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    spec = get_spec(args.model)
    probe = args.probe or str(spec.weights_subdir / "probe.npz")
    offsets = _parse_set(args.set)

    if args.model == "aria_vae":
        from latent.aria_vae_backend import AriaVAEBackend

        be = AriaVAEBackend(
            checkpoint=str(spec.primary_weight),
            tokenizer_config=str(resolve_asset(spec.tokenizer_config_local)),
            probe_path=probe,
            device_pref=args.device,
        ).load()
    else:
        from latent.cadenza_backend import CadenzaVAEBackend

        be = CadenzaVAEBackend(
            composer_ckpt=str(spec.primary_weight),
            performer_ckpt=args.performer_ckpt,
            probe_path=probe,
            device_pref=args.device,
        ).load()

    z = be.encode(args.seed) if args.seed else be.random_z()
    be.generate_with_offsets(
        z, offsets, args.out, temperature=args.temperature, top_p=args.top_p
    )
    print(f"wrote {args.out}  (offsets={offsets or 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
