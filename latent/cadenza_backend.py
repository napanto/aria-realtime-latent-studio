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


def _attrs_and_count(midi_path: str):
    """Return (n_notes, attribute_vector) for a generated MIDI."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    n = sum(len(i.notes) for i in pm.instruments if not i.is_drum)
    return n, attributes_from_midi(pm)


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
        max_len = int(getattr(self._composer.config, "max_seq_len", 192))
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
        # Cap below the Composer's positional limit. The upstream generate()
        # RoPE/KV buffer overflows by one exactly at max_seq_len, so we stay a
        # couple of tokens under it.
        ceil = int(getattr(self._composer.config, "max_seq_len", 192)) - 4
        max_steps = min(int(sampling.get("max_steps", ceil)), ceil)

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
        perturb_sigma: float = 0.8,
        samples_per_seed: int = 2,
        min_notes: int = 2,
        decode_max_steps: Optional[int] = None,
    ) -> RidgeProbe:
        """Fit z->attribute probe for Cadenza.

        For each seed we encode to ``mu`` then draw a few latents in its
        neighbourhood (``z = mu + sigma * eps``) and decode each, measuring
        attributes on the *output* MIDI (tokenizer-free ``attributes_from_midi``).
        Sampling around the posterior gives non-degenerate, varied outputs (the
        bare mean tends to collapse to a sparse skeleton), which is what makes
        the ridge fit meaningful. This learns "what direction in z moves an
        observable attribute of Cadenza's own output" — the slider semantics.
        """
        import tempfile

        rng = np.random.default_rng(0)
        max_len = int(getattr(self._composer.config, "max_seq_len", 192))
        td = Path(tmp_dir or tempfile.mkdtemp(prefix="cadenza_probe_"))
        td.mkdir(parents=True, exist_ok=True)

        mus, ys = [], []
        k = 0
        for i, p in enumerate(seed_midi_paths):
            ids = self._comp_tok.encode_midi(str(p))[:max_len]
            if len(ids) < 16:
                continue
            mu = self._encode_ids(ids)
            for s in range(max(1, samples_per_seed)):
                eps = rng.standard_normal(mu.shape).astype(np.float32)
                z = mu if s == 0 else (mu + perturb_sigma * eps)
                out = td / f"probe_{i:04d}_{s}.mid"
                dec_kw = {"temperature": 1.0}
                if decode_max_steps is not None:
                    dec_kw["max_steps"] = int(decode_max_steps)
                try:
                    self.decode(out_path=str(out), z=z, **dec_kw)
                    n_notes, y = _attrs_and_count(str(out))
                except Exception:
                    continue
                if n_notes < min_notes:
                    continue
                mus.append(z)
                ys.append(y)
                k += 1
                if k >= max_windows:
                    break
            if k >= max_windows:
                break

        if len(mus) < 16:
            raise RuntimeError(
                f"only {len(mus)} usable probe windows; supply more/denser seed "
                "MIDI (or a Performer ckpt so outputs carry velocity/pedal)."
            )
        mu_m = np.stack(mus, axis=0)
        y_m = np.stack(ys, axis=0)
        self._probe = fit_ridge_probe(mu_m, y_m, attr_names=ATTRIBUTE_NAMES)
        if save_to:
            self._probe.save(save_to)
        return self._probe
