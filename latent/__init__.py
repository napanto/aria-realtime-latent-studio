"""Latent-manipulation core (PyTorch / MPS).

The two VAE backends expose the same minimal interface so the GUI can treat
them uniformly:

  - ``encode(seed_midi_path) -> z``                 (np.ndarray, shape (z_dim,))
  - ``attribute_names -> list[str]``                 (slider labels)
  - ``direction(attr) -> w``                         (unit np.ndarray, z-space)
  - ``decode(z, out_path) -> path``                  (write a MIDI continuation)
  - ``generate_with_offsets(z, {attr: alpha}, out)`` (z' = z + Σ α·ŵ_attr)

The per-attribute *directions* come from a ridge probe fit on encoded windows
(``latent/probe.py``); for AriaVAE this mirrors ``aria_vae_latent_health.py``
exactly, for Cadenza it is the same recipe applied to Composer latents (the
upstream Cadenza code ships no probe, so we build one).
"""
from .base import LatentBackend
from .attributes import ATTRIBUTE_NAMES, attributes_from_midi

__all__ = ["LatentBackend", "ATTRIBUTE_NAMES", "attributes_from_midi"]
