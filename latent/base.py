"""Common interface + helpers for the two VAE latent backends."""
from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np


def ensure_reference_repo_on_path() -> Path:
    """Make the reference ISPR-v2 repo importable as ``src.*``.

    The VAE *model definitions* (``src/model/aria_vae.py``, ``src/model/cadenza.py``)
    and their generation helpers live in the research repo, not vendored here
    (they pull in training-only deps and are large). Set ``ISPR_V2_REPO`` to the
    repo root so ``from src.model.aria_vae import AriaVAE`` resolves.

    Falls back to a vendored ``vendor/ispr_v2_src`` if present.
    """
    candidates = []
    env = os.environ.get("ISPR_V2_REPO")
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).resolve().parent.parent / "vendor" / "ispr_v2")
    # Common sibling layout (dev machines).
    candidates.append(Path.home() / "ispr" / "ispr_v2")

    for root in candidates:
        if (root / "src" / "model" / "aria_vae.py").exists():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root

    raise ModuleNotFoundError(
        "Could not locate the ISPR-v2 reference repo (need src/model/aria_vae.py "
        "and src/model/cadenza.py). Set the ISPR_V2_REPO env var to its root, or "
        "vendor it under vendor/ispr_v2/. Searched: "
        + ", ".join(str(c) for c in candidates)
    )


def pick_device(prefer: str = "mps"):
    """Return a torch device, preferring MPS on Apple Silicon."""
    import torch

    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LatentBackend(ABC):
    """Uniform contract the GUI drives for AriaVAE and Cadenza."""

    z_dim: int
    attribute_names: tuple[str, ...]

    @abstractmethod
    def load(self) -> "LatentBackend":
        ...

    @abstractmethod
    def encode(self, seed_midi_path: str) -> np.ndarray:
        """Encode a seed MIDI window into the posterior mean ``mu`` (z_dim,)."""

    @abstractmethod
    def decode(self, z: np.ndarray, out_path: str, **sampling) -> str:
        """Decode a latent ``z`` into a MIDI continuation written to ``out_path``."""

    @abstractmethod
    def direction(self, attr: str) -> np.ndarray:
        """Unit z-space direction for ``attr`` (from the ridge probe)."""

    # -- shared --------------------------------------------------------------
    def generate_with_offsets(
        self,
        z: np.ndarray,
        offsets: dict[str, float],
        out_path: str,
        **sampling,
    ) -> str:
        """Apply ``z' = z + Σ_attr α_attr · ŵ_attr`` then decode.

        ``offsets`` maps attribute name -> α (in unit-direction multiples).
        This is exactly what a GUI slider produces.
        """
        z_prime = np.asarray(z, dtype=np.float32).copy()
        for attr, alpha in offsets.items():
            if abs(alpha) < 1e-9:
                continue
            z_prime = z_prime + float(alpha) * self.direction(attr)
        return self.decode(z_prime, out_path, **sampling)

    def random_z(self, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.z_dim).astype(np.float32)
