"""AriaVAE in MLX — a controllable latent bolted onto aria's real-time decoder.

Mirrors the torch ``src/model/aria_vae.py`` for **inference**:

  * a small bidirectional encoder maps a token window -> ``mu`` (we use ``z = mu``),
  * a prefix injector turns ``z`` into ``K=8`` decoder-space soft tokens,
  * ``K`` per-layer zero-init residual adapters add a ``z``-shift to every frozen
    decoder layer,
  * the **frozen decoder is aria's own MLX ``TransformerLM``**, untouched — we wrap
    it: the latent logic lives entirely here, so ``~/aria`` / the aria package
    never has to be edited.

Two decode paths:
  * ``decode_full(ids)`` — one-shot teacher-forced forward (parity vs torch),
  * ``prefill_prefix`` + ``decode_step`` — the KV-cached real-time path used by the
    streaming demo. The prefix occupies cache positions ``0..K-1``; real tokens
    start at ``offset = K``. Per-layer residuals are added before every block at
    every step — O(1) extra work, so the latent costs ~nothing at run time.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from aria.model import ModelConfig
from aria.inference.model_mlx import TransformerLM, apply_rotary_emb_mlx

K_PREFIX_DEFAULT = 8


# ---------------------------------------------------------------------------
# Encoder (bidirectional) — mirrors torch _BiEncoderBlock / AriaEncoder
# ---------------------------------------------------------------------------
class _BiEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.mixed_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.att_proj = nn.Linear(d_model, d_model, bias=False)
        ff = d_model * ff_mult
        self.ff_gate = nn.Linear(d_model, ff, bias=False)
        self.ff_up = nn.Linear(d_model, ff, bias=False)
        self.ff_down = nn.Linear(ff, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self._att(self.norm1(x))
        x = x + self._ff(self.norm2(x))
        return x

    def _att(self, x: mx.array) -> mx.array:
        B, S, _ = x.shape
        qkv = self.mixed_qkv(x).split(3, axis=2)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q.reshape(B, S, self.n_heads, self.d_head)
        k = k.reshape(B, S, self.n_heads, self.d_head)
        v = v.reshape(B, S, self.n_heads, self.d_head)
        q = apply_rotary_emb_mlx(q, offset=0)   # base 500000, same as decoder
        k = apply_rotary_emb_mlx(k, offset=0)
        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))   # (B,H,S,Dh)
        out = mx.fast.scaled_dot_product_attention(q=q, k=k, v=v, scale=self.scale, mask=None)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, self.n_heads * self.d_head)
        return self.att_proj(out)

    def _ff(self, x: mx.array) -> mx.array:
        return self.ff_down(nn.silu(self.ff_gate(x)) * self.ff_up(x))


class AriaEncoderMLX(nn.Module):
    def __init__(self, vocab: int, d_model: int, n_heads: int, n_layers: int,
                 ff_mult: int, z_dim: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.blocks = [_BiEncoderBlock(d_model, n_heads, ff_mult) for _ in range(n_layers)]
        self.norm = nn.LayerNorm(d_model)
        self.to_latent = nn.Linear(d_model, 2 * z_dim)
        self.z_dim = z_dim

    def __call__(self, ids: mx.array, keep: Optional[mx.array] = None) -> mx.array:
        """ids: (B,S) int -> mu: (B, z_dim). masked-mean pool over kept positions."""
        x = self.tok_emb(ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if keep is None:
            pooled = x.mean(axis=1)
        else:
            m = keep[..., None].astype(x.dtype)
            pooled = (x * m).sum(axis=1) / mx.maximum(m.sum(axis=1), 1.0)
        mu = self.to_latent(pooled)[:, : self.z_dim]
        return mu


class PrefixInjectorMLX(nn.Module):
    def __init__(self, z_dim: int, d_model: int, n_soft: int):
        super().__init__()
        self.proj = nn.Linear(z_dim, n_soft * d_model)
        self.n_soft = n_soft
        self.d_model = d_model

    def __call__(self, z: mx.array) -> mx.array:
        B = z.shape[0]
        return self.proj(z).reshape(B, self.n_soft, self.d_model)


# ---------------------------------------------------------------------------
# AriaVAE wrapper
# ---------------------------------------------------------------------------
class AriaVAEMLX:
    """Inference-only AriaVAE. Holds the frozen aria decoder + latent add-on."""

    def __init__(self, cfg: dict, dtype=mx.bfloat16):
        avc = cfg["aria_vae_config"]
        mc = cfg["decoder_model_config"]
        self.dtype = dtype
        self.z_dim = avc["z_dim"]
        self.K = avc["n_soft_tokens"]
        self.n_layers = mc["n_layers"]
        self.d_model = mc["d_model"]
        self.vocab = mc["vocab_size"]

        # --- frozen aria decoder (medium; no emb path: emb_size is None) ---
        model_config = ModelConfig(
            d_model=mc["d_model"], n_heads=mc["n_heads"], n_layers=mc["n_layers"],
            ff_mult=mc["ff_mult"], drop_p=0.0, max_seq_len=mc["max_seq_len"],
            grad_checkpoint=False,
        )
        model_config.set_vocab_size(mc["vocab_size"])
        model_config.emb_size = mc.get("emb_size")  # None for B05
        self.decoder = TransformerLM(model_config)
        self.model_config = model_config

        # --- latent add-on (encoder runs in fp32 for numeric fidelity) ---
        self.encoder = AriaEncoderMLX(
            avc["vocab_size"], avc["enc_d_model"], avc["enc_n_heads"],
            avc["enc_n_layers"], avc["enc_ff_mult"], avc["z_dim"])
        self.injector = PrefixInjectorMLX(avc["z_dim"], mc["d_model"], avc["n_soft_tokens"])
        self.z_adapters = [nn.Linear(avc["z_dim"], mc["d_model"], bias=False)
                           for _ in range(mc["n_layers"])]
        self.attr_head = [nn.Linear(avc["z_dim"], avc["z_dim"]), nn.GELU(),
                          nn.Linear(avc["z_dim"], avc["n_attrs"])]

        # set when z is chosen
        self.prefix = None       # (1,K,d) dtype
        self._residuals = None   # list[(1,1,d)] dtype (read via the property)
        # Live z control: a slider move stages a numpy z here (no MLX); the
        # decoder thread materialises prefix+residuals lazily on the next
        # `residuals` access. This keeps ALL MLX work on the single decoder
        # thread — driving MLX/Metal from two threads at once segfaults.
        self._pending_z = None
        self._z_lock = threading.Lock()

    # -- loading ----------------------------------------------------------
    @classmethod
    def load(cls, weights_dir: str, quantize: bool = False, dtype=mx.bfloat16):
        with open(os.path.join(weights_dir, "aria_vae_config.json")) as f:
            cfg = json.load(f)
        self = cls(cfg, dtype=dtype)

        dec_w = mx.load(os.path.join(weights_dir, "aria_vae_decoder.safetensors"))
        dec_w = {k: (v.astype(dtype) if v.dtype != dtype else v) for k, v in dec_w.items()}
        self.decoder.load_weights(list(dec_w.items()), strict=False)
        self.decoder.eval()
        if quantize:
            nn.quantize(self.decoder.model, group_size=32, bits=8)

        lat_w = mx.load(os.path.join(weights_dir, "aria_vae_latent.safetensors"))
        # encoder/injector/adapters kept in fp32 (parity-critical, tiny)
        self.encoder.load_weights([(k[len("encoder."):], v)
                                   for k, v in lat_w.items() if k.startswith("encoder.")],
                                  strict=False)
        self.injector.load_weights([(k[len("injector."):], v)
                                    for k, v in lat_w.items() if k.startswith("injector.")],
                                   strict=False)
        for i in range(self.n_layers):
            self.z_adapters[i].load_weights([("weight", lat_w[f"z_adapters.{i}.weight"])],
                                            strict=True)
        # attr_head (optional; only for attribute read-outs)
        ah = {k[len("attr_head."):]: v for k, v in lat_w.items() if k.startswith("attr_head.")}
        if "0.weight" in ah:
            self.attr_head[0].load_weights([("weight", ah["0.weight"]), ("bias", ah["0.bias"])], strict=True)
            self.attr_head[2].load_weights([("weight", ah["2.weight"]), ("bias", ah["2.bias"])], strict=True)
        mx.eval(self.encoder.parameters(), self.injector.parameters(),
                [a.parameters() for a in self.z_adapters])
        return self

    # -- latent -----------------------------------------------------------
    def encode(self, ids: mx.array) -> mx.array:
        """ids (1,S) int -> mu (1,z_dim) fp32."""
        return self.encoder(ids.astype(mx.int32))

    def set_z(self, z: mx.array):
        """Cache the prefix (1,K,d) and the per-layer residuals for this z.

        Drives MLX, so call it only from the thread that owns the decoder (load
        time, or the streaming thread). For a live update from another thread use
        :meth:`request_z`.
        """
        self._apply_z(z)

    def _apply_z(self, z: mx.array):
        z = z.reshape(1, self.z_dim)
        self.prefix = self.injector(z).astype(self.dtype)
        self._residuals = [self.z_adapters[i](z).reshape(1, 1, self.d_model).astype(self.dtype)
                           for i in range(self.n_layers)]
        mx.eval(self.prefix, *self._residuals)

    def request_z(self, z_np):
        """Thread-safe live z update: stage a numpy z (NO MLX work).

        The decoder thread materialises the prefix+residuals on its next
        ``residuals`` access, so MLX is never driven from two threads at once
        (which segfaults the process). Safe to call from an HTTP handler while
        the decoder streams; the change takes effect on the next token.
        """
        with self._z_lock:
            self._pending_z = np.asarray(z_np, dtype=np.float32)

    @property
    def residuals(self):
        """Per-layer z-residuals, applying any staged live z first (on THIS
        thread). Accessed once per layer per token by the decoder, so the lazy
        re-materialisation runs on the decoder thread, never the HTTP thread."""
        pending = None
        with self._z_lock:
            if self._pending_z is not None:
                pending, self._pending_z = self._pending_z, None
        if pending is not None:
            self._apply_z(mx.array(pending))
        return self._residuals

    def attrs(self, z: mx.array) -> mx.array:
        h = self.attr_head[0](z.reshape(1, self.z_dim))
        h = self.attr_head[1](h)
        return self.attr_head[2](h)

    # -- decode: one-shot teacher-forced (parity) -------------------------
    def decode_full(self, ids: mx.array) -> mx.array:
        """ids (1,S) -> logits (1,S,V) over the real-token positions only."""
        assert self.prefix is not None, "call set_z() first"
        m = self.decoder.model
        K, S = self.K, ids.shape[1]
        L = K + S
        self.decoder.setup_cache(batch_size=1, max_seq_len=max(L, 64), dtype=self.dtype)
        tok = m.tok_embeddings(ids.astype(mx.int32))                 # (1,S,d)
        x = mx.concatenate([self.prefix, tok.astype(self.dtype)], axis=1)  # (1,L,d)
        input_pos = mx.arange(L, dtype=mx.int32)
        mask = m.causal_mask[None, None, input_pos, :L]
        for li, layer in enumerate(m.encode_layers):
            x = x + self.residuals[li]
            x = layer(x, input_pos, L - 1, 0, mask)
        x = m.out_layer_norm(x)
        logits = self.decoder.lm_head(x)                             # (1,L,V)
        return logits[:, K:, :]

    # -- decode: KV-cached real-time path ---------------------------------
    def setup_stream(self, max_seq_len: int):
        self.decoder.setup_cache(batch_size=1, max_seq_len=max_seq_len, dtype=self.dtype)

    def prefill_prefix(self):
        """Write the K soft-token prefix into KV positions 0..K-1 (offset 0)."""
        assert self.prefix is not None
        m = self.decoder.model
        K = self.K
        input_pos = mx.arange(K, dtype=mx.int32)
        mask = m.causal_mask[None, None, input_pos, :K]
        x = self.prefix
        for li, layer in enumerate(m.encode_layers):
            x = x + self.residuals[li]
            x = layer(x, input_pos, K - 1, 0, mask)
        mx.eval(x)

    def forward_cached(self, idxs: mx.array, input_pos: mx.array, offset: int,
                       max_kv_pos: int) -> mx.array:
        """One/few-token cached step with per-layer residual. Returns logits (1,T,V).

        ``input_pos`` / ``offset`` are RAW positions; the K-token prefix shift is
        applied here so callers index real tokens from 0 (mirrors EMBEDDING_OFFSET).
        """
        m = self.decoder.model
        K = self.K
        ip = input_pos + K
        off = offset + K
        x = m.tok_embeddings(idxs.astype(mx.int32)).astype(self.dtype)
        mask = m.causal_mask[None, None, ip, : max_kv_pos + K + 1]
        for li, layer in enumerate(m.encode_layers):
            x = x + self.residuals[li]
            x = layer(x, ip, max_kv_pos + K, off, mask)
        x = m.out_layer_norm(x)
        return self.decoder.lm_head(x)
