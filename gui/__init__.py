"""Local web GUI (FastAPI + vanilla JS).

Stack rationale (see README "GUI stack choice"): a local web app keeps the heavy
compute (MLX) in Python while giving us zero-build controls — a model selector,
transport controls, and sampling knobs — without bundling a native toolkit. Both
Aria models are driven by the Python real-time engine, which owns the Core MIDI
ports through mido/rtmidi and streams continuations to a chosen output.
"""
