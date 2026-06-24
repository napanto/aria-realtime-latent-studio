#!/usr/bin/env python3
"""Fetch all model weights from the Hugging Face Hub into ``weights/``.

Reads the same :data:`models.registry.MODEL_REGISTRY` the app uses, so the
download wiring can never drift from the runtime wiring. Weights are gitignored
and never committed.

Auth: reads ``HF_TOKEN`` from the environment (the repos are mostly public, but
a token avoids rate limits and is required if any repo is private). No token is
ever hardcoded.

Usage:
    HF_TOKEN=hf_xxx python scripts/download_models.py            # all 4 models
    python scripts/download_models.py --only aria_jazz aria_vae  # a subset
    python scripts/download_models.py --list                     # show status
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.registry import MODEL_REGISTRY, ModelSpec  # noqa: E402
from studio import WEIGHTS_DIR  # noqa: E402


def _hf_download(spec: ModelSpec, token: str | None) -> None:
    from huggingface_hub import hf_hub_download

    spec.weights_subdir.mkdir(parents=True, exist_ok=True)
    for f in spec.files:
        dest = spec.local_path(f)
        if dest.exists():
            print(f"  [skip] {spec.key}/{f.local_name} (exists)")
            continue
        print(f"  [get ] {f.repo_id}:{f.path_in_repo} -> {dest}")
        cached = hf_hub_download(
            repo_id=f.repo_id,
            repo_type=f.repo_type,
            filename=f.path_in_repo,
            token=token,
        )
        # hf_hub_download returns a path inside the HF cache; symlink/copy it to
        # our flat weights/<key>/<local_name> layout for predictable paths.
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            os.symlink(cached, dest)
        except OSError:
            import shutil

            shutil.copy2(cached, dest)


def _print_status() -> None:
    print(f"weights dir: {WEIGHTS_DIR}")
    for key, spec in MODEL_REGISTRY.items():
        status = "READY" if spec.is_downloaded() else "missing"
        print(f"  [{status:7}] {key:12} ({spec.backend.value}) — {spec.display_name}")
        for f in spec.files:
            mark = "x" if spec.local_path(f).exists() else " "
            print(f"      [{mark}] {f.local_name}  <- {f.repo_id}:{f.path_in_repo}")
    print(
        "\nNOTE: cadenza_vae downloads the Composer only — the Performer fill "
        "checkpoint is not published on HF (see STATUS.md). The Cadenza backend "
        "falls back to composition-skeleton decode until a Performer .pt is "
        "supplied via CadenzaVAEBackend(performer_ckpt=...)."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", help="subset of model keys to fetch")
    ap.add_argument("--list", action="store_true", help="show download status")
    args = ap.parse_args()

    if args.list:
        _print_status()
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "[warn] HF_TOKEN not set; proceeding unauthenticated "
            "(fine for public repos, may hit rate limits).",
            file=sys.stderr,
        )

    keys = args.only or list(MODEL_REGISTRY)
    for key in keys:
        if key not in MODEL_REGISTRY:
            print(f"[err ] unknown model key {key!r}", file=sys.stderr)
            return 2
        spec = MODEL_REGISTRY[key]
        print(f"== {key} ({spec.display_name}) ==")
        _hf_download(spec, token)

    print("\nDone.")
    _print_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
