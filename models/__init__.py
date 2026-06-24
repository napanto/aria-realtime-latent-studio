"""Model-abstraction layer.

Four backends behind one registry:

  ============  =========  ===================  ===============================
  key           backend    runtime              role
  ============  =========  ===================  ===============================
  aria_base     MLX        realtime engine      original Aria (loubb)
  aria_jazz     MLX        realtime engine      our jazz fine-tuned Aria
  aria_vae      torch/MPS  latent-manip engine  AriaVAE (frozen Aria + latent)
  cadenza_vae   torch/MPS  latent-manip engine  Cadenza Composer+Performer VAE
  ============  =========  ===================  ===============================

The two *plain Aria* models are real-time AR transformers driven by the
proven MLX demo (``realtime/``). The two *VAE* models are PyTorch and run on
MPS (or CPU) through the latent-manipulation engine (``latent/``); they expose
the per-attribute latent sliders.

``MODEL_REGISTRY`` is the single source of truth for which checkpoint /
tokenizer / backend each model key uses. ``scripts/download_models.py`` reads
the same HF spec so the wiring never drifts.
"""
from .registry import MODEL_REGISTRY, ModelSpec, Backend, get_spec

__all__ = ["MODEL_REGISTRY", "ModelSpec", "Backend", "get_spec"]
