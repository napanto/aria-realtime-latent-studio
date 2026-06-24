"""Closed-form ridge probe: z -> attributes, and its columns as directions.

This is the *exact* recipe from the reference repo's
``src/aria_vae_latent_health.py`` (the "independent ridge probe, head-free"
block). Given a stack of encoded latents ``mu`` (N, z_dim) and matching
attribute targets ``Y`` (N, n_attrs), it solves

    W = (Xc^T Xc + λI)^{-1} Xc^T (Y - ȳ)        # W : (z_dim, n_attrs)

Each **column** ``W[:, i]`` is the z-space direction that most increases
attribute ``i``. We normalise it to a unit vector and use it as the slider
direction:  ``z' = z + α · ŵ_attr``.

The probe weights are *not* persisted anywhere upstream (only R² is), so we
fit them here from a small set of seed windows and cache the result to
``weights/<model>/probe.npz`` for instant reuse.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .attributes import ATTRIBUTE_NAMES


@dataclass
class RidgeProbe:
    W: np.ndarray            # (z_dim, n_attrs) — columns are directions
    x_mean: np.ndarray       # (z_dim,)
    y_mean: np.ndarray       # (n_attrs,)
    attr_names: tuple[str, ...]
    r2: dict[str, float]

    @property
    def z_dim(self) -> int:
        return int(self.W.shape[0])

    def direction(self, attr: str, normalize: bool = True) -> np.ndarray:
        """Unit z-space direction for ``attr`` (the probe column)."""
        i = self.attr_names.index(attr)
        w = self.W[:, i].astype(np.float64)
        n = np.linalg.norm(w)
        if normalize and n > 1e-12:
            w = w / n
        return w.astype(np.float32)

    def predict(self, mu: np.ndarray) -> np.ndarray:
        return (mu.astype(np.float64) - self.x_mean) @ self.W + self.y_mean

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W=self.W,
            x_mean=self.x_mean,
            y_mean=self.y_mean,
            attr_names=np.asarray(self.attr_names),
            r2=np.asarray([self.r2[a] for a in self.attr_names], dtype=np.float64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RidgeProbe":
        d = np.load(path, allow_pickle=True)
        names = tuple(str(x) for x in d["attr_names"])
        return cls(
            W=d["W"],
            x_mean=d["x_mean"],
            y_mean=d["y_mean"],
            attr_names=names,
            r2={n: float(r) for n, r in zip(names, d["r2"])},
        )


def fit_ridge_probe(
    mu: np.ndarray,            # (N, z_dim)
    Y: np.ndarray,            # (N, n_attrs)
    *,
    lam: float = 1.0,
    attr_names: tuple[str, ...] = ATTRIBUTE_NAMES,
) -> RidgeProbe:
    """Fit the head-free ridge probe (80/20 split for the reported R²)."""
    X = np.asarray(mu, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n_w = X.shape[0]
    if n_w < 16:
        raise ValueError(
            f"need >=16 windows to fit a probe, got {n_w}. "
            "Point fit at more/longer seed MIDI."
        )
    ntr = min(max(X.shape[1] // 2, int(0.8 * n_w)), n_w - 8)

    xm = X[:ntr].mean(0, keepdims=True)
    ym = Y[:ntr].mean(0, keepdims=True)
    Xc = X - xm

    A = Xc[:ntr].T @ Xc[:ntr] + lam * np.eye(X.shape[1])
    W = np.linalg.solve(A, Xc[:ntr].T @ (Y[:ntr] - ym))   # (z_dim, n_attrs)

    # Held-out R² per attribute.
    pred = Xc[ntr:] @ W + ym
    yte = Y[ntr:]
    ss_res = ((yte - pred) ** 2).sum(0)
    ss_tot = np.maximum(((yte - yte.mean(0, keepdims=True)) ** 2).sum(0), 1e-9)
    rr = 1.0 - ss_res / ss_tot
    r2 = {name: float(rr[i]) for i, name in enumerate(attr_names)}

    return RidgeProbe(
        W=W,
        x_mean=xm.reshape(-1),
        y_mean=ym.reshape(-1),
        attr_names=attr_names,
        r2=r2,
    )
