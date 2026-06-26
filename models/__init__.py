"""Model-abstraction layer.

Two MLX Aria backends behind one registry:

  ==========  =======  ===============  ===========================
  key         backend  runtime          role
  ==========  =======  ===============  ===========================
  aria_base   MLX      realtime engine  original Aria (loubb)
  aria_jazz   MLX      realtime engine  our jazz fine-tuned Aria
  ==========  =======  ===============  ===========================

Both models are real-time AR transformers driven by the proven MLX demo
(``realtime/``) with grammar-constrained decoding.

``MODEL_REGISTRY`` is the single source of truth for which checkpoint /
tokenizer / backend each model key uses. ``scripts/download_models.py`` reads
the same HF spec so the wiring never drifts.
"""
from .registry import MODEL_REGISTRY, ModelSpec, Backend, get_spec

__all__ = ["MODEL_REGISTRY", "ModelSpec", "Backend", "get_spec"]
