"""Runtime latent control for the real-time studio.

Loads the fitted ``latent_directions.npz`` and turns MIDI-CC slider values into
**calibrated** moves of the latent ``z``. For attribute ``k`` with ridge column
``w_k`` (z -> attribute), the minimal-norm move that shifts the probe's predicted
attribute by ``Δ`` units is ``z += (Δ / ||w_k||²) · w_k``. A CC (0..127, centred
at 64) maps linearly to ``Δ ∈ [-range_k, +range_k]`` with ``range_k = gain·σ_k``,
so a full slider sweep changes that attribute by ±``gain``·std — intuitive and
disentangled-ish (each axis is the probe direction for one attribute).

Also supports raw per-dimension traversal of the highest-variance latent dims
(unlabelled but always available, even where a probe is weak).
"""
from __future__ import annotations

import numpy as np

# attributes whose probe is too weak / null to expose as a slider (||w||~0 or low R²)
_MIN_WNORM = 1e-3


class LatentController:
    def __init__(self, directions_path: str, gain_sigma: float = 2.0, min_r2: float = 0.4):
        d = np.load(directions_path, allow_pickle=True)
        self.W = d["W"].astype(np.float64)                # (z_dim, n_attr)
        self.names = [str(x) for x in d["names"]]
        self.attr_std = d["attr_std"].astype(np.float64)
        self.mu_mean = d["mu_mean"].astype(np.float64)
        self.mu_std = d["mu_std"].astype(np.float64)
        self.r2 = d["r2"].astype(np.float64)
        self.z_dim = self.W.shape[0]
        self.gain = gain_sigma
        # precompute per-attr move vector for +1 unit of Δ: w_k / ||w_k||²
        self.wnorm2 = (self.W ** 2).sum(0)               # (n_attr,)
        self.unit_move = np.zeros_like(self.W)
        for k in range(self.W.shape[1]):
            if self.wnorm2[k] > _MIN_WNORM:
                self.unit_move[:, k] = self.W[:, k] / self.wnorm2[k]
        # which attributes are usable as sliders
        self.active = [k for k in range(len(self.names))
                       if self.wnorm2[k] > _MIN_WNORM and (np.isnan(self.r2[k]) or self.r2[k] >= min_r2)]
        self.base = np.zeros(self.z_dim)                 # set from an encoded seed
        self.offsets = {}                                # attr_idx -> Δ (attribute units)
        self.dim_offsets = {}                            # dim -> value (in σ units)

    # -- base latent ------------------------------------------------------
    def set_base(self, z):
        self.base = np.asarray(z, np.float64).reshape(self.z_dim)

    def attr_index(self, name: str) -> int:
        return self.names.index(name)

    # -- CC -> Δ ----------------------------------------------------------
    def cc_to_delta(self, attr_idx: int, cc_value: int) -> float:
        """CC 0..127 (centre 64) -> Δ in raw attribute units (±gain·σ)."""
        frac = (int(cc_value) - 64) / 64.0               # [-1, 1]
        return frac * self.gain * float(self.attr_std[attr_idx])

    def set_cc(self, attr_idx: int, cc_value: int):
        self.offsets[attr_idx] = self.cc_to_delta(attr_idx, cc_value)

    def set_attr_delta(self, attr_idx: int, delta: float):
        self.offsets[attr_idx] = float(delta)

    def set_dim_cc(self, dim: int, cc_value: int, sigma_range: float = 3.0):
        frac = (int(cc_value) - 64) / 64.0
        self.dim_offsets[dim] = frac * sigma_range * float(self.mu_std[dim])

    def clear(self):
        self.offsets.clear(); self.dim_offsets.clear()

    # -- compose ----------------------------------------------------------
    def z(self) -> np.ndarray:
        """Current latent = base + Σ attribute moves + Σ per-dim moves."""
        z = self.base.copy()
        for k, delta in self.offsets.items():
            z = z + delta * self.unit_move[:, k]
        for d, val in self.dim_offsets.items():
            z[d] += val
        return z.astype(np.float32)

    # -- diagnostics ------------------------------------------------------
    def predicted_attrs(self, z=None) -> dict:
        z = self.base if z is None else np.asarray(z, np.float64)
        pred = (z - self.mu_mean) @ self.W            # centred readout (Δ from mean)
        return {self.names[k]: float(pred[k]) for k in range(len(self.names))}

    def slider_report(self) -> str:
        items = []
        for k in self.active:
            items.append(f"{self.names[k]}(r2={self.r2[k]:.2f})")
        return "controllable: " + ", ".join(items)
