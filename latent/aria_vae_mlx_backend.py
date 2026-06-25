"""AriaVAE latent backend — real-time MLX path (Apple Silicon).

This is the MLX counterpart of :mod:`latent.aria_vae_backend` (torch/MPS). It
fulfils STATUS TODO #6: instead of torch full-sequence re-eval per token, it
injects the ``z``-prefix (K=8 soft tokens) into the **KV cache** of the frozen
real-time Aria MLX decoder and adds a per-layer zero-init ``z``-residual, so
latent-conditioned generation runs at real-time MLX speed (~52 tok/s / 19 ms
per token on a MacBook Air M1).

  encode : seed MIDI -> Aria AbsTokenizer ids -> bidirectional encoder -> mu (128,)
  decode : z -> injector (8 soft prefix tokens, cached) + per-layer residual ->
           frozen Aria MLX decoder -> grammar-constrained KV-cached AR
           continuation -> MIDI
  direction : unit-normalised column of the fitted ridge probe
              (``latent_directions.npz``, 320 PiJAMA windows — TODO #7)

The validated engine modules live (vendored, byte-identical to the parity-
checked originals) under ``studio/mlx_vae/``. They use flat imports, so this
backend prepends that directory to ``sys.path`` before importing — mirroring how
``latent.base.ensure_reference_repo_on_path`` makes the research ``src.*``
package importable for the torch backends.

Requires ``mlx`` + the EleutherAI ``aria`` + ``ariautils`` packages, which are
Apple-Silicon-only. On Linux the import of the engine fails — that is expected
and is why this backend is kept separate from the torch one.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .attributes import ATTRIBUTE_NAMES
from .base import LatentBackend


class _ProbeShim:
    """Minimal probe view the GUI reads (``backend._probe.r2`` dict +
    ``probe_ready``). The MLX backends keep directions in
    ``latent_directions*.npz``; this exposes their per-attribute R²."""

    def __init__(self, names, r2):
        self.r2 = {str(n): (float(v) if np.isfinite(v) else 0.0)
                   for n, v in zip(names, np.asarray(r2))}


def _ensure_engine_on_path() -> Path:
    """Make the vendored ``studio/mlx_vae`` engine importable (flat imports).

    Also ensures the EleutherAI ``aria`` package is importable: the MLX engine
    imports ``aria.model`` / ``aria.inference.model_mlx`` (the frozen real-time
    decoder). It is provided by the ``aria`` package on Apple Silicon, or via the
    ``ARIA_REPO`` env var pointing at a checkout.
    """
    engine_dir = Path(__file__).resolve().parent.parent / "studio" / "mlx_vae"
    if str(engine_dir) not in sys.path:
        sys.path.insert(0, str(engine_dir))
    aria_repo = os.environ.get("ARIA_REPO")
    if aria_repo and aria_repo not in sys.path:
        sys.path.insert(0, aria_repo)
    return engine_dir


class AriaVAEMLXBackend(LatentBackend):
    z_dim = 128
    attribute_names = ATTRIBUTE_NAMES

    def __init__(
        self,
        weights_dir: str,
        tokenizer_config: str,
        *,
        directions_path: Optional[str] = None,
        aria_repo: Optional[str] = None,
        seq_len: int = 512,
        quantize: bool = False,
        gain_sigma: float = 2.0,
    ):
        """
        Parameters
        ----------
        weights_dir
            Directory holding the MLX safetensors + config:
            ``aria_vae_config.json``, ``aria_vae_decoder.safetensors``,
            ``aria_vae_latent.safetensors`` (and, ideally,
            ``latent_directions.npz``).
        tokenizer_config
            Path to the Aria ``AbsTokenizer`` config JSON (the demo/jazz
            tokenizer; vocab 2675, pedal on).
        directions_path
            Fitted ``latent_directions.npz`` for the sliders. Defaults to
            ``<weights_dir>/latent_directions.npz`` if present.
        aria_repo
            Optional path to an ``aria`` package checkout (else import ``aria``
            from the environment / ``ARIA_REPO``).
        """
        self.weights_dir = weights_dir
        self.tokenizer_config = tokenizer_config
        self.directions_path = directions_path or str(
            Path(weights_dir) / "latent_directions.npz"
        )
        self.aria_repo = aria_repo
        self.seq_len = seq_len
        self.quantize = quantize
        self.gain_sigma = gain_sigma

        self._model = None          # AriaVAEMLX
        self._tok = None            # ariautils AbsTokenizer
        self._grammar = None        # build_grammar(tok) dict
        self._ctrl = None           # LatentController (probe directions)
        self._probe = None          # _ProbeShim (GUI R² view)
        self._eos_id = None
        self._last_prompt_ids: Optional[np.ndarray] = None
        self._mx = None             # mlx.core handle (set in load)

    # -- loading -------------------------------------------------------------
    def load(self) -> "AriaVAEMLXBackend":
        if self.aria_repo:
            os.environ.setdefault("ARIA_REPO", self.aria_repo)
        _ensure_engine_on_path()

        import mlx.core as mx
        from aria_vae_mlx import AriaVAEMLX
        from grammar import build_grammar
        from latent_control import LatentController
        from ariautils.tokenizer import AbsTokenizer

        self._mx = mx
        self._model = AriaVAEMLX.load(self.weights_dir, quantize=self.quantize)
        self._tok = AbsTokenizer(config_path=self.tokenizer_config)
        self._eos_id = self._tok.tok_to_id.get(
            getattr(self._tok, "eos_tok", "<E>")
        )
        self._grammar = build_grammar(self._tok)

        if Path(self.directions_path).exists():
            self._ctrl = LatentController(
                self.directions_path, gain_sigma=self.gain_sigma
            )
            self._probe = _ProbeShim(self._ctrl.names, self._ctrl.r2)
        return self

    # -- contract ------------------------------------------------------------
    def encode(self, seed_midi_path: str) -> np.ndarray:
        from ariautils.midi import MidiDict

        ids = self._tok.encode(
            self._tok.tokenize(MidiDict.from_midi(str(seed_midi_path)))
        )[: self.seq_len]
        self._last_prompt_ids = np.asarray(ids, dtype=np.int32)
        mu = self._model.encode(
            self._mx.array(self._last_prompt_ids[None, :])
        )
        self._mx.eval(mu)
        return np.asarray(mu)[0].astype(np.float32)

    def decode(self, z: np.ndarray, out_path: str, **sampling) -> str:
        from generate_latent import generate_one, sanitize_aria_tokens

        temperature = float(sampling.get("temperature", 1.0))
        min_p = float(sampling.get("min_p", 0.03))
        n_new = int(sampling.get("max_new_tokens", sampling.get("n_new", 384)))
        constrained = bool(sampling.get("constrained", True))
        prompt_ids = sampling.get("prompt_ids")
        if prompt_ids is None:
            prompt_ids = self._last_prompt_ids
            if prompt_ids is None:
                raise RuntimeError(
                    "no seed prompt available; call encode(seed) before decode "
                    "or pass prompt_ids= in sampling"
                )
        prompt_ids = list(np.asarray(prompt_ids, dtype=np.int32))

        # Cache the z-prefix + per-layer residuals for this latent.
        self._model.set_z(self._mx.array(np.asarray(z, dtype=np.float32)))

        full, _tps = generate_one(
            self._model,
            self._tok,
            self._grammar,
            prompt_ids,
            n_new,
            temperature,
            min_p,
            self._eos_id,
            constrained,
        )
        toks = sanitize_aria_tokens(self._tok.decode(full), self._tok)
        md = self._tok.detokenize(toks)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        md.to_midi().save(str(out_path))
        return str(out_path)

    def direction(self, attr: str) -> np.ndarray:
        """Unit-normalised z-direction for ``attr`` from the ridge probe.

        The probe column ``W[:, k]`` (z -> attribute) is normalised to a unit
        vector so that the ``alpha`` in ``generate_with_offsets`` matches the
        torch backend's convention (alpha = move in unit-direction multiples).
        """
        if self._ctrl is None:
            raise RuntimeError(
                "no latent directions loaded; pass directions_path= or place "
                "latent_directions.npz in the weights dir. AriaVAE-MLX "
                "directions come from the fitted ridge probe columns."
            )
        k = self._ctrl.attr_index(attr)
        w = self._ctrl.W[:, k].astype(np.float64)
        n = np.linalg.norm(w)
        if n > 1e-12:
            w = w / n
        return w.astype(np.float32)
