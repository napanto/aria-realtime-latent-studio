"""AriaVAE latent backend (PyTorch / MPS).

Wraps the reference repo's ``src.model.aria_vae`` + ``src.aria_vae_generate``:

  encode : seed MIDI -> Aria tokens -> encoder -> mu (128,)
  decode : mu -> injector -> 8 soft prefix tokens -> frozen Aria decoder
           -> grammar-constrained AR continuation -> MIDI
  probe  : ridge fit of mu -> 7 attributes over seed windows; columns are the
           per-attribute directions for z' = z + alpha * w_attr

The decoder *is* the real-time jazz Aria, so this path produces in-style jazz
continuations whose performance attributes the sliders steer.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np

from .attributes import ATTRIBUTE_NAMES
from .base import LatentBackend, ensure_reference_repo_on_path, pick_device
from .probe import RidgeProbe, fit_ridge_probe


class AriaVAEBackend(LatentBackend):
    z_dim = 128
    attribute_names = ATTRIBUTE_NAMES

    def __init__(
        self,
        checkpoint: str,
        tokenizer_config: str,
        *,
        probe_path: Optional[str] = None,
        device_pref: str = "mps",
        seq_len: int = 512,
        aria_repo: Optional[str] = None,
    ):
        self.checkpoint = checkpoint
        self.tokenizer_config = tokenizer_config
        self.probe_path = probe_path
        self.device_pref = device_pref
        self.seq_len = seq_len
        self.aria_repo = aria_repo
        self._model = None
        self._cfg = None
        self._tok = None
        self._device = None
        self._probe: Optional[RidgeProbe] = None
        self._gen = None  # the imported src.aria_vae_generate module
        self._av = None   # the imported src.model.aria_vae module
        self._last_prompt_ids: Optional[np.ndarray] = None

    # -- loading -----------------------------------------------------------
    def load(self) -> "AriaVAEBackend":
        ensure_reference_repo_on_path()
        import torch
        from src import aria_vae_generate as gen      # type: ignore
        from src.model import aria_vae as av          # type: ignore

        self._gen = gen
        self._av = av
        self._device = pick_device(self.device_pref)

        self._model, self._cfg = gen.load_aria_vae(self.checkpoint, self._device)
        self._model.eval()
        self._tok = gen.load_tokenizer(self.tokenizer_config, self.aria_repo)

        if self.probe_path and Path(self.probe_path).exists():
            self._probe = RidgeProbe.load(self.probe_path)
        return self

    # -- autocast ----------------------------------------------------------
    def _autocast(self):
        import torch

        dev = self._device
        if dev.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        # MPS/CPU: bf16 autocast is unreliable across torch versions -> fp32.
        return nullcontext()

    def _encode_ids(self, ids: np.ndarray) -> np.ndarray:
        """Run the encoder over one window of Aria token ids -> mu (128,)."""
        import torch

        t = torch.from_numpy(np.asarray(ids, dtype=np.int64)).view(1, -1)
        t = t.to(self._device)
        with torch.no_grad(), self._autocast():
            mu, _logvar = self._model.encoder(t)
        return mu.float().cpu().numpy().reshape(-1)

    # -- public API --------------------------------------------------------
    def encode(self, seed_midi_path: str) -> np.ndarray:
        ids = self._tokenize_midi(seed_midi_path)[: self.seq_len]
        # Remember the prompt window: its head carries the
        # ("prefix","instrument","piano") + <S> tokens that AbsTokenizer.
        # detokenize() asserts on. decode() reuses it to seed the continuation.
        self._last_prompt_ids = ids
        return self._encode_ids(ids)

    def _piano_prefix_ids(self) -> np.ndarray:
        """A minimal valid prompt head for the random-z (no-seed) case.

        Mirrors the start of ``tok.tokenize``: the piano instrument prefix
        followed by BOS, so ``detokenize`` finds exactly one instrument.
        """
        head = [("prefix", "instrument", "piano"), self._tok.bos_tok]
        return np.asarray(self._tok.encode(head), dtype=np.int64)

    def _tokenize_midi(self, midi_path: str) -> np.ndarray:
        from ariautils.midi import MidiDict

        md = MidiDict.from_midi(str(midi_path))
        md.remove_redundant_pedals()
        toks = self._tok.tokenize(md, add_dim_tok=False)
        # Drop the trailing EOS if present (encoder wants the raw window).
        if self._tok.eos_tok in toks:
            toks = toks[: toks.index(self._tok.eos_tok)]
        ids = np.asarray(self._tok.encode(toks), dtype=np.int64)
        return ids

    def decode(self, z: np.ndarray, out_path: str, **sampling) -> str:
        import torch
        from ariautils.midi import MidiDict  # noqa: F401

        temperature = float(sampling.get("temperature", 0.95))
        top_p = float(sampling.get("top_p", 0.9))
        max_new = int(sampling.get("max_new_tokens", 512))
        prompt_ids = sampling.get("prompt_ids")  # optional conditioning window

        z_t = torch.from_numpy(np.asarray(z, dtype=np.float32)).view(1, -1)
        z_t = z_t.to(self._device)

        with torch.no_grad(), self._autocast():
            prefix = self._model.injector(z_t)         # (1, K, d_model)

        if prompt_ids is None:
            # Prefer the last encoded seed window (it carries the piano-prefix
            # head detokenize asserts on); else a minimal valid piano prefix.
            prompt_ids = getattr(self, "_last_prompt_ids", None)
            if prompt_ids is None:
                prompt_ids = self._piano_prefix_ids()
        prompt_ids = np.asarray(prompt_ids, dtype=np.int64)
        prompt_t = torch.from_numpy(prompt_ids)
        prompt_t = prompt_t.to(self._device)

        grammar = self._gen.GrammarFSM(self._gen.build_grammar(self._tok, self._device))
        z_for_decode = (
            z_t if getattr(self._cfg, "z_resid_adapter", False) else None
        )
        emitted = self._gen.generate_continuation(
            self._model,
            prompt_t,
            prefix,
            max_new_tokens=max_new,
            temperature=temperature,
            top_p=top_p,
            eos_id=self._tok.tok_to_id.get(self._tok.eos_tok),
            pad_id=self._tok.tok_to_id.get(self._tok.pad_tok),
            generator=None,
            autocast_ctx=self._autocast,
            z=z_for_decode,
            grammar=grammar,
        )

        full_ids = list(prompt_ids) + list(emitted)
        toks = self._tok.decode(full_ids)
        toks = self._gen.sanitize_aria_tokens(toks, self._tok)
        md = self._tok.detokenize(toks)
        md.to_midi().save(str(out_path))
        return str(out_path)

    def direction(self, attr: str) -> np.ndarray:
        if self._probe is None:
            raise RuntimeError(
                "no ridge probe loaded; call build_probe(seed_dir) or pass "
                "probe_path. AriaVAE directions come from the probe columns."
            )
        return self._probe.direction(attr)

    # -- probe construction ------------------------------------------------
    def build_probe(
        self,
        seed_midi_paths: list[str],
        *,
        window: int = 512,
        stride: int = 256,
        max_windows: int = 400,
        save_to: Optional[str] = None,
    ) -> RidgeProbe:
        """Fit the z->attribute ridge probe from a set of seed MIDIs.

        Uses the reference repo's token-based ``compute_attributes`` for the
        targets (the same function the upstream health report uses), so the
        directions match ``aria_vae_latent_health.py`` exactly.
        """
        import torch
        from src.model.aria_vae import compute_attributes  # type: ignore

        mus, ys = [], []
        for p in seed_midi_paths:
            try:
                ids = self._tokenize_midi(p)   # may raise on non-piano/empty MIDI
            except Exception:
                continue
            n = len(ids)
            for start in range(0, max(1, n - 32), stride):
                w = ids[start : start + window]
                if len(w) < 32:
                    continue
                try:
                    mu = self._encode_ids(w)
                    w_t = torch.from_numpy(w.astype(np.int64))
                    y = compute_attributes(w_t, self._tok).numpy().reshape(-1)
                except Exception:
                    continue
                if not (np.isfinite(mu).all() and np.isfinite(y).all()):
                    continue
                mus.append(mu)
                ys.append(y)
                if len(mus) >= max_windows:
                    break
            if len(mus) >= max_windows:
                break

        if len(mus) < 16:
            raise RuntimeError(
                f"only {len(mus)} usable AriaVAE probe windows; supply more "
                "piano MIDI seeds."
            )
        mu_m = np.stack(mus, axis=0)
        y_m = np.stack(ys, axis=0)
        self._probe = fit_ridge_probe(mu_m, y_m, attr_names=ATTRIBUTE_NAMES)
        if save_to:
            self._probe.save(save_to)
        return self._probe
