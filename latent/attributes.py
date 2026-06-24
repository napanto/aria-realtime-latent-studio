"""Performance attributes used as latent-slider axes.

These are the seven interpretable attributes the AriaVAE ridge probe regresses
against (``src/model/aria_vae.py::ATTRIBUTE_NAMES`` in the reference repo). We
re-declare them here so the GUI labels and the probe stay in lock-step, and we
provide a tokenizer-agnostic extractor that reads attributes straight from a
MIDI file (via ``pretty_midi``). The MIDI-based extractor is what we use to
(a) build a probe for Cadenza and (b) *measure* the effect of a slider move
(encode the generated output, recompute attributes, check it shifted).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Order is load-bearing: it must match the AriaVAE probe column order.
ATTRIBUTE_NAMES: tuple[str, ...] = (
    "velocity_mean",   # mean MIDI velocity (0..127)
    "velocity_std",    # std MIDI velocity
    "note_density",    # notes per second
    "ioi_entropy",     # Shannon entropy (nats) of inter-onset-interval hist
    "pitch_mean",      # mean MIDI pitch (0..127)
    "pitch_std",       # std MIDI pitch
    "pedal_fraction",  # fraction of time the sustain pedal is held [0..1]
)
N_ATTRS = len(ATTRIBUTE_NAMES)

# Human-friendly slider labels + a sensible default range (std-units of α).
ATTR_LABELS: dict[str, str] = {
    "velocity_mean": "Loudness (mean velocity)",
    "velocity_std": "Dynamics (velocity spread)",
    "note_density": "Note density",
    "ioi_entropy": "Rhythmic complexity (IOI entropy)",
    "pitch_mean": "Register (mean pitch)",
    "pitch_std": "Pitch spread",
    "pedal_fraction": "Pedal usage",
}


def _ioi_entropy(onsets_s: list[float], n_bins: int = 16) -> float:
    """Shannon entropy (nats) of the log-spaced inter-onset-interval hist."""
    if len(onsets_s) < 3:
        return 0.0
    iois = np.diff(np.sort(np.asarray(onsets_s, dtype=np.float64)))
    iois = iois[iois > 1e-4]
    if iois.size < 2:
        return 0.0
    logs = np.log(iois)
    hist, _ = np.histogram(logs, bins=n_bins)
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


@dataclass
class _Notes:
    pitches: np.ndarray
    velocities: np.ndarray
    onsets_s: np.ndarray
    span_s: float
    pedal_held_s: float


def _collect_notes_pretty_midi(midi) -> _Notes:
    import pretty_midi  # noqa: F401  (typing only)

    pitches, vels, onsets = [], [], []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            pitches.append(n.pitch)
            vels.append(n.velocity)
            onsets.append(n.start)

    if not onsets:
        return _Notes(np.array([]), np.array([]), np.array([]), 0.0, 0.0)

    onsets = np.asarray(onsets, dtype=np.float64)
    span = float(max(midi.get_end_time(), onsets.max()) - onsets.min())
    span = max(span, 1e-3)

    # Sustain pedal (CC64) held-time across all instruments.
    pedal_held = 0.0
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        down_t: Optional[float] = None
        for cc in sorted(
            (c for c in inst.control_changes if c.number == 64),
            key=lambda c: c.time,
        ):
            if cc.value >= 64 and down_t is None:
                down_t = cc.time
            elif cc.value < 64 and down_t is not None:
                pedal_held += max(0.0, cc.time - down_t)
                down_t = None
        if down_t is not None:
            pedal_held += max(0.0, midi.get_end_time() - down_t)

    return _Notes(
        np.asarray(pitches, dtype=np.float64),
        np.asarray(vels, dtype=np.float64),
        onsets,
        span,
        pedal_held,
    )


def attributes_from_midi(midi_or_path) -> np.ndarray:
    """Compute the 7 attributes from a MIDI file path or a PrettyMIDI object.

    Returns a float32 vector in ``ATTRIBUTE_NAMES`` order. Tokenizer-free, so
    it works for both AriaVAE and Cadenza outputs.
    """
    import pretty_midi

    midi = (
        midi_or_path
        if isinstance(midi_or_path, pretty_midi.PrettyMIDI)
        else pretty_midi.PrettyMIDI(str(midi_or_path))
    )
    nt = _collect_notes_pretty_midi(midi)
    if nt.pitches.size == 0:
        return np.zeros(N_ATTRS, dtype=np.float32)

    vel_mean = float(nt.velocities.mean())
    vel_std = float(nt.velocities.std())
    density = float(nt.pitches.size / nt.span_s)
    ioi_ent = _ioi_entropy(nt.onsets_s.tolist())
    pitch_mean = float(nt.pitches.mean())
    pitch_std = float(nt.pitches.std())
    pedal_frac = float(min(1.0, nt.pedal_held_s / nt.span_s))

    return np.asarray(
        [vel_mean, vel_std, density, ioi_ent, pitch_mean, pitch_std, pedal_frac],
        dtype=np.float32,
    )
