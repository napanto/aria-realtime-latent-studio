"""Validated MLX VAE inference engine (vendored).

These modules are the **parity-verified** MLX port of the AriaVAE / Cadenza VAE
inference paths, copied verbatim from the source research repo. They were
validated on the real macOS host (MacBook Air M1):

  * AriaVAE: mu-parity 2.4e-7 vs torch, decode 100% argmax-identical, ~52 tok/s
    (19 ms/token) KV-cached real-time generation.
  * Cadenza two-stage: 100% argmax-identical, latent control note_density +0.94.

The modules use flat top-level imports (``from aria_vae_mlx import ...``) so they
also run as standalone CLI scripts. The studio's latent backends
(:mod:`latent.aria_vae_mlx_backend`, :mod:`latent.cadenza_mlx_backend`) put this
directory on ``sys.path`` before importing, keeping the vendored files
byte-identical to the validated originals.

Importing the heavy modules requires ``mlx`` + the EleutherAI ``aria`` package
(Apple-Silicon only), so this package marker imports nothing eagerly.
"""
