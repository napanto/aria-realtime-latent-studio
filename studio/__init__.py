"""aria-realtime-studio: shared package root.

Holds the model registry and the paths to the local ``weights/`` directory
populated by ``scripts/download_models.py``.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repo root = parent of this package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = Path(os.environ.get("LATENT_STUDIO_WEIGHTS", REPO_ROOT / "weights"))
ASSETS_DIR = REPO_ROOT / "assets"
SEED_MIDI_DIR = ASSETS_DIR / "seed_midi"

__all__ = ["REPO_ROOT", "WEIGHTS_DIR", "ASSETS_DIR", "SEED_MIDI_DIR"]
