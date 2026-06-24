"""Cadenza VAE latent backend (PyTorch / MPS).

Two-stage: Composer (composition tokens, z-conditioned) -> insert MASK at
Velocity/MicroTime/Pedal slots -> Performer fill -> MIDI.

  encode : seed MIDI -> PerTok *composition* ids -> Composer.encode -> mu (128,)
  decode : mu -> Composer.generate -> performer fill -> PrettyMIDI
  probe  : built here (upstream Cadenza ships NO probe). We regress encoded
           Composer latents against attributes measured from the two-stage
           output MIDI. Columns are the per-attribute directions.

Because the Composer latent governs *compositional* structure (the Performer
adds dynamics/microtiming), some performance attributes (velocity_*) are only
weakly steerable from z — the probe R² surfaces this honestly per attribute.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np

from .attributes import ATTRIBUTE_NAMES, attributes_from_midi
from .base import LatentBackend, ensure_reference_repo_on_path, pick_device
from .probe import RidgeProbe, fit_ridge_probe


class CadenzaVAEBackend(LatentBackend):
    z_dim = 128
    attribute_names = ATTRIBUTE_NAMES

    def __init__(
        self,
        composer_ckpt: str,
        performer_ckpt: Optional[str] = None,
        *,
        probe_path: Optional[str] = None,
        device_pref: str = "mps",
        cache_root: Optional[str] = None,
    ):
        self.composer_ckpt = composer_ckpt
        self.performer_ckpt = performer_ckpt
        self.probe_path = probe_path
        self.device_pref = device_pref
        self.cache_root = cache_root
        self._composer = None
        self._performer = None
        self._comp_tok = None
        self._perf_tok = None
        self._device = None
        self._probe: Optional[RidgeProbe] = None
        self._ts = None  # imported src.cadenza_two_stage_generate module

    def load(self) -> "CadenzaVAEBackend":
        ensure_reference_repo_on_path()
        import torch
        from src import cadenza_two_stage_generate as ts   # type: ignore
        from src.data.pertok_tokenizer import PerTokWrapper  # type: ignore

        self._ts = ts
        self._device = pick_device(self.device_pref)

        self._comp_tok = PerTokWrapper.from_default(
            cache_root=self.cache_root, mode="composition"
        )
        self._perf_tok = PerTokWrapper.from_default(
            cache_root=None, mode="performance"
        )

        self._composer = ts._load_composer(
            self.composer_ckpt,
            self._device,
            vocab_size=self._comp_tok.vocab_size,
            pad_id=int(self._comp_tok.pad_id),
            bos_id=int(self._comp_tok.bos_id),
            eos_id=int(self._comp_tok.eos_id),
        )
        if self.performer_ckpt and Path(self.performer_ckpt).exists():
            self._performer = ts._load_performer(self.performer_ckpt, self._device)

        if self.probe_path and Path(self.probe_path).exists():
            self._probe = RidgeProbe.load(self.probe_path)
        return self

    def _autocast(self):
        import torch

        if self._device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _encode_ids(self, ids: np.ndarray) -> np.ndarray:
        import torch

        t = torch.from_numpy(np.asarray(ids, dtype=np.int64)).view(1, -1)
        t = t.to(self._device)
        attn = (t != int(self._comp_tok.pad_id)).long()
        with torch.no_grad(), self._autocast():
            mu = self._composer.encode(t, attention_mask=attn, sample=False)
        return mu.float().cpu().numpy().reshape(-1)

    def encode(self, seed_midi_path: str) -> np.ndarray:
        ids = self._comp_tok.encode_midi(str(seed_midi_path))
        max_len = int(getattr(self._composer.cfg, "max_seq_len", 192))
        return self._encode_ids(ids[:max_len])

    def decode(self, z: np.ndarray, out_path: str, **sampling) -> str:
        """Composer.generate(z) -> performer fill -> MIDI.

        If no Performer checkpoint was provided, falls back to decoding the
        Composer (composition) tokens directly to MIDI (no performance fill),
        with a note printed by the caller / STATUS.md.
        """
        import torch

        temperature = float(sampling.get("temperature", 1.0))
        top_k = int(sampling.get("top_k", 0))
        top_p = float(sampling.get("top_p", 1.0))
        max_steps = int(sampling.get("max_steps", 192))

        z_t = torch.from_numpy(np.asarray(z, dtype=np.float32)).view(1, -1)
        z_t = z_t.to(self._device)

        with torch.no_grad(), self._autocast():
            gen_ids = self._composer.generate(
                z_t,
                max_steps=max_steps,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        gen_np = gen_ids[0].detach().cpu().numpy().astype(np.int32)
        # Trim at EOS / trailing pad (mirrors cadenza_two_stage_generate.main).
        eos = int(self._comp_tok.eos_id)
        eos_pos = np.where(gen_np == eos)[0]
        if eos_pos.size:
            gen_np = gen_np[: int(eos_pos[0])]
        pad = int(self._comp_tok.pad_id)
        nonpad = np.where(gen_np != pad)[0]
        if nonpad.size:
            gen_np = gen_np[: int(nonpad[-1]) + 1]

        if self._performer is not None:
            # Two-stage: insert MASKs + Performer fill -> PrettyMIDI written.
            self._ts._performer_fill_and_write(
                gen_np,
                comp_tok=self._comp_tok,
                perf_tok=self._perf_tok,
                performer=self._performer,
                device=self._device,
                dtype=torch.float32,
                performer_sample=bool(sampling.get("performer_sample", False)),
                out_gen_path=Path(out_path),
            )
        else:
            # Fallback: decode the composition skeleton directly.
            midi = self._comp_tok.decode(gen_np.tolist())
            midi.write(str(out_path))
        return str(out_path)

    def direction(self, attr: str) -> np.ndarray:
        if self._probe is None:
            raise RuntimeError(
                "no Cadenza ridge probe loaded; call build_probe(seed_dir) "
                "or pass probe_path. Cadenza has no upstream probe."
            )
        return self._probe.direction(attr)

    def build_probe(
        self,
        seed_midi_paths: list[str],
        *,
        max_windows: int = 200,
        save_to: Optional[str] = None,
        tmp_dir: Optional[str] = None,
    ) -> RidgeProbe:
        """Fit z->attribute probe for Cadenza.

        We encode each seed window to mu, decode it back through the two-stage
        pipeline, and measure attributes on the *output* MIDI (tokenizer-free
        ``attributes_from_midi``). This learns "what direction in z moves this
        observable attribute of Cadenza's own output" — the right notion for a
        controllability slider.
        """
        import tempfile

        max_len = int(getattr(self._composer.cfg, "max_seq_len", 192))
        td = Path(tmp_dir or tempfile.mkdtemp(prefix="cadenza_probe_"))
        td.mkdir(parents=True, exist_ok=True)

        mus, ys = [], []
        for i, p in enumerate(seed_midi_paths):
            ids = self._comp_tok.encode_midi(str(p))[:max_len]
            if len(ids) < 16:
                continue
            mu = self._encode_ids(ids)
            out = td / f"probe_{i:04d}.mid"
            try:
                self.decode(mu, str(out), temperature=1.0, max_steps=max_len)
                y = attributes_from_midi(str(out))
            except Exception:
                continue
            mus.append(mu)
            ys.append(y)
            if len(mus) >= max_windows:
                break

        if len(mus) < 16:
            raise RuntimeError(
                f"only {len(mus)} usable probe windows; supply more seed MIDI."
            )
        mu_m = np.stack(mus, axis=0)
        y_m = np.stack(ys, axis=0)
        self._probe = fit_ridge_probe(mu_m, y_m, attr_names=ATTRIBUTE_NAMES)
        if save_to:
            self._probe.save(save_to)
        return self._probe
