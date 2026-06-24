"""Local web GUI (FastAPI + vanilla JS + WebMIDI).

Stack rationale (see README "GUI stack choice"): a local web app keeps the
heavy compute (MLX / torch-MPS) in Python while giving us zero-build sliders,
a model selector, transport controls, and — via the browser's WebMIDI API — a
uniform way to enumerate and route Core MIDI devices on macOS without bundling
a native toolkit. The two plain-Aria models are driven by the Python real-time
engine (which owns the MIDI ports through mido/rtmidi); the two VAE models are
driven by the latent engine, where each slider maps to one attribute direction.
"""
