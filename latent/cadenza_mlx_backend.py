"""Cadenza VAE latent backend — MLX two-stage path (Apple Silicon).

The MLX counterpart of :mod:`latent.cadenza_backend` (torch). It fulfils STATUS
TODO #5 (the Performer is now published + recreated, so the two-stage render is
unblocked) using the parity-checked MLX Composer + Performer:

  encode : seed MIDI -> PerTok-p ids -> Composer encoder -> mu (128,)
  decode : z -> Composer KV-cached AR (performance tokens) -> mask every
           Velocity/MicroTime/Pedal slot -> Performer bidirectional fill ->
           PerTok-p detok -> MIDI
  direction : unit-normalised column of the fitted Cadenza ridge probe
              (``latent_directions_cadenza.npz``, 320 PiJAMA windows)

Parity vs torch (host, M1, fp32): Composer mu 5.3e-6, decode 100% argmax,
Performer fill 100% argmax. Latent control (slider -> generated attribute):
note_density +0.94 — stronger than AriaVAE because the Cadenza decoder is NOT
frozen. Offline (~2.7 s / 256-token clip), so it is on-demand regeneration, not
live streaming (same framing as the torch backend).

Needs ``mlx`` + ``miditok`` + ``symusic`` (Apple-Silicon side); the engine
modules are vendored under ``studio/mlx_vae``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .attributes import ATTRIBUTE_NAMES
from .base import LatentBackend

_PERFORMANCE_PREFIXES = ("Velocity_", "MicroTiming_", "Pedal_", "PedalOff_")


def _ensure_engine_on_path() -> Path:
    engine_dir = Path(__file__).resolve().parent.parent / "studio" / "mlx_vae"
    if str(engine_dir) not in sys.path:
        sys.path.insert(0, str(engine_dir))
    # the PerTok wrapper is vendored at the repo root as src/data/pertok_tokenizer.py
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return engine_dir


class CadenzaVAEMLXBackend(LatentBackend):
    z_dim = 128
    attribute_names = ATTRIBUTE_NAMES

    def __init__(
        self,
        weights_dir: str,
        *,
        directions_path: Optional[str] = None,
        ispr_path: Optional[str] = None,
        prompt_len: int = 384,
        max_generate_steps: int = 512,
        performer_sample: bool = False,
        gain_sigma: float = 2.0,
    ):
        """
        Parameters
        ----------
        weights_dir
            Dir with ``cadenza_{composer,performer}.safetensors`` +
            ``cadenza_config.json`` (and ideally
            ``latent_directions_cadenza.npz``). The recreated Performer is at
            HF ``vae_campaign/A05_kongFT/performer_recreated/`` — convert it with
            ``studio/mlx_vae/convert_cadenza.py``.
        ispr_path
            Path whose ``src/data/pertok_tokenizer.py`` provides the PerTok-p
            wrapper. Defaults to the repo's vendored copy.
        """
        self.weights_dir = weights_dir
        self.directions_path = directions_path or str(
            Path(weights_dir) / "latent_directions_cadenza.npz"
        )
        self.ispr_path = ispr_path
        self.prompt_len = prompt_len
        self.max_generate_steps = max_generate_steps
        self.performer_sample = performer_sample
        self.gain_sigma = gain_sigma

        self._mx = None
        self._composer = None
        self._performer = None
        self._tok = None
        self._perf_id_set = None
        self._ctrl = None

    def load(self) -> "CadenzaVAEMLXBackend":
        _ensure_engine_on_path()
        import mlx.core as mx
        from cadenza_mlx import CadenzaComposerMLX, CadenzaPerformerMLX
        from cadenza_two_stage_mlx import load_perf_tok
        from latent_control import LatentController

        self._mx = mx
        self._composer = CadenzaComposerMLX.load(self.weights_dir, dtype=mx.float32)
        self._performer = CadenzaPerformerMLX.load(self.weights_dir, dtype=mx.float32)
        ispr = self.ispr_path or str(Path(__file__).resolve().parent.parent)
        self._tok = load_perf_tok(ispr)
        self._perf_id_set = set(self._tok.token_ids_by_prefix(_PERFORMANCE_PREFIXES))
        if Path(self.directions_path).exists():
            self._ctrl = LatentController(self.directions_path, gain_sigma=self.gain_sigma)
        return self

    def encode(self, seed_midi_path: str) -> np.ndarray:
        ids = self._tok.encode_midi(str(seed_midi_path))
        ids = np.asarray(ids[: min(self.prompt_len, self._composer.cfg["max_seq_len"])],
                         dtype=np.int32)
        if ids.size < 8:
            raise RuntimeError(f"seed produced only {ids.size} tokens")
        mu = self._composer.encode(self._mx.array(ids[None, :]))
        self._mx.eval(mu)
        return np.asarray(mu)[0].astype(np.float32)

    def decode(self, z: np.ndarray, out_path: str, **sampling) -> str:
        mx = self._mx
        temperature = float(sampling.get("temperature", 1.0))
        top_k = int(sampling.get("top_k", 0))
        top_p = float(sampling.get("top_p", 1.0))
        seed = int(sampling.get("seed", 0))

        gen = self._composer.generate(
            mx.array(np.asarray(z, np.float32)), max_steps=self.max_generate_steps,
            temperature=temperature, top_k=top_k, top_p=top_p, key=mx.random.key(seed))
        mx.eval(gen)
        gen_ids = np.array(gen[0]).astype(np.int32)
        eos_id, pad_id = int(self._tok.eos_id), int(self._tok.pad_id)
        eos_pos = np.where(gen_ids == eos_id)[0]
        if eos_pos.size:
            gen_ids = gen_ids[: int(eos_pos[0])]
        nonpad = np.where(gen_ids != pad_id)[0]
        if nonpad.size:
            gen_ids = gen_ids[: int(nonpad[-1]) + 1]
        if gen_ids.size < 4:
            raise RuntimeError("Composer emitted too few tokens")

        # two-stage: mask expressive slots, Performer fill
        mask_id = int(self._tok.mask_id)
        perf_in = gen_ids.copy()
        mask_positions = np.array([t in self._perf_id_set for t in gen_ids], dtype=bool)
        perf_in[mask_positions] = mask_id
        plen = min(perf_in.size, self._performer.max_seq_len)
        perf_in, mask_positions = perf_in[:plen], mask_positions[:plen]
        plogits = self._performer.fill(mx.array(perf_in[None, :]))
        mx.eval(plogits)
        plogits_np = np.array(plogits[0])
        out_ids = perf_in.copy()
        midx = np.where(mask_positions)[0]
        if self.performer_sample and midx.size:
            for p in midx:
                row = plogits_np[p] / max(temperature, 1e-9)
                row = row - row.max()
                pr = np.exp(row); pr /= pr.sum()
                out_ids[p] = int(np.random.choice(len(pr), p=pr))
        elif midx.size:
            out_ids[midx] = plogits_np[midx].argmax(-1).astype(np.int32)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        self._tok.decode(out_ids).write(str(out_path))
        return str(out_path)

    def direction(self, attr: str) -> np.ndarray:
        if self._ctrl is None:
            raise RuntimeError(
                "no Cadenza latent directions loaded; place "
                "latent_directions_cadenza.npz in the weights dir "
                "(fit via studio/mlx_vae/fit_latent_directions_cadenza.py).")
        k = self._ctrl.attr_index(attr)
        w = self._ctrl.W[:, k].astype(np.float64)
        n = np.linalg.norm(w)
        if n > 1e-12:
            w = w / n
        return w.astype(np.float32)
