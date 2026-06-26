"""MLX decoding helpers for the real-time Aria engine.

Holds :mod:`grammar` — the note/pedal grammar FSM the real-time decoder uses for
grammar-constrained decoding of the Aria ``AbsTokenizer`` stream (every sampled
token is masked to a grammatically valid next category, so detokenize never
fails mid-stream).

The module uses a flat top-level import (``from grammar import ...``); the
real-time engine (:mod:`realtime.aria_demo_mlx`) prepends this directory to
``sys.path`` before importing it. ``grammar`` needs only ``mlx`` + ``numpy``, so
this package marker imports nothing eagerly.
"""
