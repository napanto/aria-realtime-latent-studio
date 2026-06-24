"""Real-time MLX continuation engine.

Thin, importable wrapper around the **proven** EleutherAI/aria real-time demo
(vendored verbatim as ``realtime/aria_demo_mlx.py``). The two plain-Aria
backends (original Aria, jazz fine-tuned Aria) run through this engine
unchanged — only the checkpoint path + tokenizer config differ.

We deliberately do NOT re-implement the demo's careful low-latency path
(KV-cache, chunked prefill, duration recalculation, beam-of-3 first onset,
min-p sampling, MIDI scheduling). We import it and parameterise it.
"""
from .engine import RealtimeAriaEngine, RealtimeConfig

__all__ = ["RealtimeAriaEngine", "RealtimeConfig"]
