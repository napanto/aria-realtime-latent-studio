"""Cadenza two-stage VAE in MLX — offline latent manipulation for jazz piano.

Mirrors the torch ``src/model/cadenza.py`` (Composer) and
``src/model/cadenza_performer.py`` (Performer) for **inference** on Apple MLX:

  * **CadenzaComposerMLX** — a transformer VAE. A bidirectional RoPE encoder
    pools position-0 of the final hidden state → (μ, logσ²); we use z = μ. A
    causal RoPE decoder reconstructs/generates tokens. The latent is injected
    two ways, both paper/checkpoint-faithful:
      1. a SINGLE shared ``W_pre(z)`` broadcast-added to every decoder block's
         hidden state BEFORE self-attention (paper §3.1 Eq. 5), and
      2. per-block AdaLN-zero FiLM (``z_adaln=True``): ``adaln(z)`` →
         (shift1,scale1,shift2,scale2), modulating ln1/ln2 outputs as
         ``h·(1+scale)+shift``. The final ``adaln`` Linear was zero-init at
         train start; the trained gates are loaded from the checkpoint.
    The lm_head is weight-tied to the token embedding.

  * **CadenzaPerformerMLX** — a BERT-style bidirectional encoder (sinusoidal
    PE, post-LN, GELU, tied out_proj) that re-fills MASK positions (velocity /
    microtime / pedal) given the rest of a PerTok-p sequence.

Both decode paths the studio needs:
  * Composer ``decode_full(ids)`` — one-shot teacher-forced forward (parity),
  * Composer ``generate(z, ...)`` — KV-cached autoregressive sampling,
  * Performer ``fill(ids)`` — one bidirectional forward → per-position logits.

The Composer here was trained on the *performance* PerTok cache (vocab 220),
so it emits Velocity/MicroTime/Pedal tokens directly; the Performer re-fills
those slots when the two-stage pipeline blanks them.
"""
from __future__ import annotations

import json
import math
import os
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# RoPE — byte-for-byte the torch convention in src/model/cadenza.py:
#   _rope_freqs: angles = pos ⊗ inv_freq, then cat([angles, angles], -1)
#   _apply_rope: x1,x2 = chunk(2); x*cos + cat([-x2, x1])*sin
# ---------------------------------------------------------------------------


def _rope_tables(d_head: int, max_seq: int, base: float = 10000.0):
    """Return (sin, cos) each shape (max_seq, d_head), matching torch."""
    inv_freq = 1.0 / (base ** (mx.arange(0, d_head, 2).astype(mx.float32) / d_head))
    pos = mx.arange(max_seq).astype(mx.float32)
    angles = mx.outer(pos, inv_freq)                       # (S, d_head/2)
    angles = mx.concatenate([angles, angles], axis=-1)     # (S, d_head)
    return mx.sin(angles), mx.cos(angles)


def _apply_rope(x: mx.array, sin: mx.array, cos: mx.array) -> mx.array:
    """x: (B, H, L, Dh).  sin/cos: (L, Dh) broadcast over (B, H)."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


def _sinusoidal_pe(max_len: int, d_model: int) -> mx.array:
    """Standard Vaswani 2017 sinusoidal PE — matches torch ``_sinusoidal_pe``."""
    pe = mx.zeros((max_len, d_model))
    position = mx.arange(0, max_len).astype(mx.float32).reshape(max_len, 1)
    div_term = mx.exp(mx.arange(0, d_model, 2).astype(mx.float32)
                      * (-math.log(10000.0) / d_model))
    arg = position * div_term                              # (max_len, d_model/2)
    s = mx.sin(arg)
    c = mx.cos(arg)
    # interleave: pe[:,0::2]=sin, pe[:,1::2]=cos
    pe = mx.zeros((max_len, d_model))
    idx_even = mx.arange(0, d_model, 2)
    idx_odd = mx.arange(1, d_model, 2)
    pe[:, idx_even] = s
    pe[:, idx_odd] = c
    return pe


# ===========================================================================
# Composer
# ===========================================================================


class _ComposerEncoderBlock(nn.Module):
    """Bidirectional, RoPE, LayerNorm-pre, GELU FF. Matches EncoderBlock."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff0 = nn.Linear(d_model, d_ff)
        self.ff3 = nn.Linear(d_ff, d_model)

    def __call__(self, x, sin, cos, key_padding_mask):
        x = x + self._att(self.ln1(x), sin, cos, key_padding_mask)
        x = x + self.ff3(nn.gelu(self.ff0(self.ln2(x))))
        return x

    def _att(self, x, sin, cos, key_padding_mask):
        B, L, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, L, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        q = _apply_rope(q, sin, cos)
        k = _apply_rope(k, sin, cos)
        mask = None
        if key_padding_mask is not None:
            # (B, 1, 1, L): -inf at PAD positions.
            mask = mx.where(key_padding_mask.reshape(B, 1, 1, L),
                            mx.array(-mx.inf, dtype=q.dtype),
                            mx.array(0.0, dtype=q.dtype))
        out = mx.fast.scaled_dot_product_attention(q=q, k=k, v=v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.n_heads * self.d_head)
        return self.attn_out(out)


class _ComposerDecoderBlock(nn.Module):
    """Causal, RoPE, LayerNorm-pre, GELU FF, shared W_pre add + optional AdaLN.

    Matches DecoderBlock.forward:
        x = x + x_pre
        if z_adaln:  shift1,scale1,shift2,scale2 = adaln(z)
                     attn(modulate(ln1(x), shift1, scale1)); x += attn
                     x += ff(modulate(ln2(x), shift2, scale2))
        else:        attn(ln1(x)); x += attn ; x += ff(ln2(x))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 z_adaln: bool, latent_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.d_model = d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff0 = nn.Linear(d_model, d_ff)
        self.ff3 = nn.Linear(d_ff, d_model)
        self.z_adaln = z_adaln
        if z_adaln:
            # adaln.1 is the (4*d_model, latent_dim) Linear; adaln.0 is SiLU.
            self.adaln1 = nn.Linear(latent_dim, 4 * d_model)

    @staticmethod
    def _modulate(h, shift, scale):
        return h * (1.0 + scale) + shift

    def __call__(self, x, x_pre, sin, cos, z, kv_cache=None, return_cache=False,
                 start_pos: int = 0):
        x = x + x_pre
        if self.z_adaln:
            mod = self.adaln1(nn.silu(z))                # (B, 4d)
            shift1, scale1, shift2, scale2 = mx.split(mod, 4, axis=-1)
            shift1 = shift1[:, None, :]; scale1 = scale1[:, None, :]
            shift2 = shift2[:, None, :]; scale2 = scale2[:, None, :]
            attn_out, new_kv = self._att(self._modulate(self.ln1(x), shift1, scale1),
                                         sin, cos, kv_cache, return_cache, start_pos)
            x = x + attn_out
            x = x + self.ff3(nn.gelu(self.ff0(self._modulate(self.ln2(x), shift2, scale2))))
        else:
            attn_out, new_kv = self._att(self.ln1(x), sin, cos, kv_cache,
                                         return_cache, start_pos)
            x = x + attn_out
            x = x + self.ff3(nn.gelu(self.ff0(self.ln2(x))))
        return x, new_kv

    def _att(self, x, sin, cos, kv_cache, return_cache, start_pos):
        B, L_new, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, L_new, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, L_new, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, L_new, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        q = _apply_rope(q, sin, cos)
        k = _apply_rope(k, sin, cos)
        if kv_cache is not None:
            K_past, V_past = kv_cache
            k = mx.concatenate([K_past, k], axis=2)
            v = mx.concatenate([V_past, v], axis=2)
            mask = None                                  # single new token, attends to all past
        else:
            mask = "causal" if L_new > 1 else None
        out = mx.fast.scaled_dot_product_attention(q=q, k=k, v=v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L_new, self.n_heads * self.d_head)
        new_kv = (k, v) if return_cache else None
        return self.attn_out(out), new_kv


class CadenzaComposerMLX:
    """Inference-only Cadenza Composer VAE in MLX."""

    def __init__(self, cfg: dict, dtype=mx.float32):
        self.cfg = cfg
        self.dtype = dtype
        self.d_model = cfg["d_model"]
        self.latent_dim = cfg["latent_dim"]
        self.vocab = cfg["vocab_size"]
        self.n_enc = cfg["n_layers_enc"]
        self.n_dec = cfg["n_layers_dec"]
        self.n_heads_enc = cfg["n_heads_enc"]
        self.n_heads_dec = cfg["n_heads_dec"]
        self.z_adaln = bool(cfg.get("z_adaln", False))
        self.pad_id = cfg["pad_id"]
        self.bos_id = cfg["bos_id"]
        self.eos_id = cfg["eos_id"]
        base = float(cfg.get("rope_base", 10000.0))
        max_seq = cfg["max_seq_len"]

        self.tok_embed = nn.Embedding(self.vocab, self.d_model)
        self.enc_blocks = [_ComposerEncoderBlock(self.d_model, self.n_heads_enc, cfg["d_ff_enc"])
                           for _ in range(self.n_enc)]
        self.enc_ln_out = nn.LayerNorm(self.d_model)
        self.mu_head = nn.Linear(self.d_model, self.latent_dim)
        self.logvar_head = nn.Linear(self.d_model, self.latent_dim)
        self.W_pre = nn.Linear(self.latent_dim, self.d_model, bias=False)
        self.dec_blocks = [_ComposerDecoderBlock(self.d_model, self.n_heads_dec,
                                                 cfg["d_ff_dec"], self.z_adaln, self.latent_dim)
                           for _ in range(self.n_dec)]
        self.dec_ln_out = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab, bias=False)

        d_head_enc = self.d_model // self.n_heads_enc
        d_head_dec = self.d_model // self.n_heads_dec
        self.enc_sin, self.enc_cos = _rope_tables(d_head_enc, max_seq, base)
        self.dec_sin, self.dec_cos = _rope_tables(d_head_dec, max_seq, base)

    # -- loading ----------------------------------------------------------
    def load_weights(self, w: dict):
        emap = {}
        for k, v in w.items():
            emap[k] = v.astype(self.dtype) if v.dtype != self.dtype else v

        def grab(prefix):
            return [(k[len(prefix):], v) for k, v in emap.items() if k.startswith(prefix)]

        self.tok_embed.load_weights([("weight", emap["tok_embed.weight"])], strict=True)
        self.enc_ln_out.load_weights(grab("enc_ln_out."), strict=True)
        self.mu_head.load_weights(grab("mu_head."), strict=True)
        self.logvar_head.load_weights(grab("logvar_head."), strict=True)
        self.W_pre.load_weights([("weight", emap["W_pre.weight"])], strict=True)
        self.dec_ln_out.load_weights(grab("dec_ln_out."), strict=True)
        self.lm_head.load_weights([("weight", emap["lm_head.weight"])], strict=True)

        for i, blk in enumerate(self.enc_blocks):
            b = f"enc_blocks.{i}."
            blk.ln1.load_weights(grab(b + "ln1."), strict=True)
            blk.ln2.load_weights(grab(b + "ln2."), strict=True)
            blk.qkv.load_weights([("weight", emap[b + "attn.qkv.weight"])], strict=True)
            blk.attn_out.load_weights([("weight", emap[b + "attn.out.weight"])], strict=True)
            blk.ff0.load_weights([("weight", emap[b + "ff.0.weight"]),
                                  ("bias", emap[b + "ff.0.bias"])], strict=True)
            blk.ff3.load_weights([("weight", emap[b + "ff.3.weight"]),
                                  ("bias", emap[b + "ff.3.bias"])], strict=True)
        for i, blk in enumerate(self.dec_blocks):
            b = f"dec_blocks.{i}."
            blk.ln1.load_weights(grab(b + "ln1."), strict=True)
            blk.ln2.load_weights(grab(b + "ln2."), strict=True)
            blk.qkv.load_weights([("weight", emap[b + "attn.qkv.weight"])], strict=True)
            blk.attn_out.load_weights([("weight", emap[b + "attn.out.weight"])], strict=True)
            blk.ff0.load_weights([("weight", emap[b + "ff.0.weight"]),
                                  ("bias", emap[b + "ff.0.bias"])], strict=True)
            blk.ff3.load_weights([("weight", emap[b + "ff.3.weight"]),
                                  ("bias", emap[b + "ff.3.bias"])], strict=True)
            if self.z_adaln:
                blk.adaln1.load_weights([("weight", emap[b + "adaln.1.weight"]),
                                         ("bias", emap[b + "adaln.1.bias"])], strict=True)
        self._eval_all()

    def _eval_all(self):
        mods = [self.tok_embed, self.enc_ln_out, self.mu_head, self.logvar_head,
                self.W_pre, self.dec_ln_out, self.lm_head, *self.enc_blocks, *self.dec_blocks]
        mx.eval([m.parameters() for m in mods])

    @classmethod
    def load(cls, weights_dir: str, dtype=mx.float32):
        with open(os.path.join(weights_dir, "cadenza_config.json")) as f:
            cfg = json.load(f)["composer"]
        self = cls(cfg, dtype=dtype)
        w = mx.load(os.path.join(weights_dir, "cadenza_composer.safetensors"))
        self.load_weights(w)
        return self

    # -- encode -----------------------------------------------------------
    def encode(self, ids: mx.array) -> mx.array:
        """ids (B, L) int -> mu (B, latent_dim). Pools position 0."""
        ids = ids.astype(mx.int32)
        B, L = ids.shape
        key_padding_mask = (ids == self.pad_id)            # (B, L) True at PAD
        h = self.tok_embed(ids)
        sin = self.enc_sin[:L].astype(h.dtype)
        cos = self.enc_cos[:L].astype(h.dtype)
        for blk in self.enc_blocks:
            h = blk(h, sin, cos, key_padding_mask)
        h = self.enc_ln_out(h)
        pooled = h[:, 0, :]
        mu = self.mu_head(pooled)
        return mu

    # -- decode: one-shot teacher-forced (parity) -------------------------
    def decode_full(self, z: mx.array, dec_in: mx.array) -> mx.array:
        """z (B, latent), dec_in (B, L) int -> logits (B, L, V)."""
        B, L = dec_in.shape
        h = self.tok_embed(dec_in.astype(mx.int32))
        x_pre = self.W_pre(z)[:, None, :].astype(h.dtype)
        sin = self.dec_sin[:L].astype(h.dtype)
        cos = self.dec_cos[:L].astype(h.dtype)
        for blk in self.dec_blocks:
            h, _ = blk(h, x_pre, sin, cos, z, kv_cache=None, return_cache=False)
        h = self.dec_ln_out(h)
        return self.lm_head(h)

    # -- generate: KV-cached AR sampling ----------------------------------
    def generate(self, z: mx.array, max_steps: int = 512, temperature: float = 1.0,
                 top_k: int = 0, top_p: float = 1.0, key=None) -> mx.array:
        """Autoregressive decode from z. Returns (B, <=max_steps) ids
        excluding the BOS prime. Mirrors torch Cadenza.generate."""
        B = z.shape[0]
        if key is None:
            key = mx.random.key(0)
        x_pre = self.W_pre(z)[:, None, :].astype(self.dtype)

        def step(dec_in, caches, start_pos):
            L_new = dec_in.shape[1]
            h = self.tok_embed(dec_in.astype(mx.int32)).astype(self.dtype)
            sin = self.dec_sin[start_pos:start_pos + L_new].astype(h.dtype)
            cos = self.dec_cos[start_pos:start_pos + L_new].astype(h.dtype)
            new_caches = []
            for i, blk in enumerate(self.dec_blocks):
                kv = None if caches is None else caches[i]
                h, nkv = blk(h, x_pre, sin, cos, z, kv_cache=kv, return_cache=True,
                             start_pos=start_pos)
                new_caches.append(nkv)
            h = self.dec_ln_out(h)
            return self.lm_head(h), new_caches

        dec_in = mx.full((B, 1), self.bos_id, dtype=mx.int32)
        logits, caches = step(dec_in, None, 0)
        last = logits[:, -1, :]
        emitted = []
        alive = mx.ones((B,), dtype=mx.bool_)
        pos = 1
        for _ in range(max_steps):
            key, sk = mx.random.split(key)
            nxt = self._sample(last, temperature, top_k, top_p, sk)   # (B,)
            nxt = mx.where(alive, nxt, mx.full((B,), self.pad_id, dtype=mx.int32))
            emitted.append(nxt[:, None])
            alive = alive & (nxt != self.eos_id)
            mx.eval(alive)
            if not bool(mx.any(alive).item()):
                break
            logits, caches = step(nxt[:, None], caches, pos)
            last = logits[:, -1, :]
            pos += 1
        if not emitted:
            return mx.zeros((B, 0), dtype=mx.int32)
        return mx.concatenate(emitted, axis=1)

    def _sample(self, logits, temperature, top_k, top_p, key) -> mx.array:
        if temperature <= 0:
            return mx.argmax(logits, axis=-1).astype(mx.int32)
        logits = logits / max(temperature, 1e-9)
        V = logits.shape[-1]
        if top_k and 0 < top_k < V:
            kth = mx.sort(logits, axis=-1)[:, -top_k][:, None]
            logits = mx.where(logits < kth, mx.array(-mx.inf, dtype=logits.dtype), logits)
        if 0.0 < top_p < 1.0:
            order = mx.argsort(logits, axis=-1)[:, ::-1]               # descending
            sl = mx.take_along_axis(logits, order, axis=-1)
            probs = mx.softmax(sl, axis=-1)
            cum = mx.cumsum(probs, axis=-1)
            cutoff = cum > top_p
            # keep at least one
            cutoff[:, 0] = False
            sl = mx.where(cutoff, mx.array(-mx.inf, dtype=sl.dtype), sl)
            # scatter back
            unsorted = mx.zeros_like(logits)
            inv = mx.argsort(order, axis=-1)
            logits = mx.take_along_axis(sl, inv, axis=-1)
        return mx.random.categorical(logits, key=key).astype(mx.int32)


# ===========================================================================
# Performer
# ===========================================================================


class _PerformerLayer(nn.Module):
    """Post-LN BERT encoder layer (matches nn.TransformerEncoderLayer,
    norm_first=False, GELU). Bidirectional self-attention, no RoPE."""

    def __init__(self, d_model: int, nhead: int, dim_ff: int):
        super().__init__()
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.scale = self.d_head ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def __call__(self, x, key_padding_mask):
        # Post-LN: x = norm1(x + attn(x)); x = norm2(x + ff(x)).
        a = self._att(x, key_padding_mask)
        x = self.norm1(x + a)
        f = self.linear2(nn.gelu(self.linear1(x)))
        x = self.norm2(x + f)
        return x

    def _att(self, x, key_padding_mask):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.nhead, self.d_head).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.nhead, self.d_head).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.nhead, self.d_head).transpose(0, 2, 1, 3)
        mask = None
        if key_padding_mask is not None:
            mask = mx.where(key_padding_mask.reshape(B, 1, 1, L),
                            mx.array(-mx.inf, dtype=q.dtype),
                            mx.array(0.0, dtype=q.dtype))
        out = mx.fast.scaled_dot_product_attention(q=q, k=k, v=v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.nhead * self.d_head)
        return self.out_proj(out)


class CadenzaPerformerMLX:
    """Inference-only Cadenza Performer (BERT-style fill) in MLX."""

    def __init__(self, cfg: dict, dtype=mx.float32):
        self.cfg = cfg
        self.dtype = dtype
        self.d_model = cfg["d_model"]
        self.vocab = cfg["vocab_size"]
        self.nhead = cfg["nhead"]
        self.n_layers = cfg["n_layers"]
        self.pad_id = cfg["pad_id"]
        self.mask_id = cfg["mask_id"]
        self.max_seq_len = cfg["max_seq_len"]
        self.tok_embed = nn.Embedding(self.vocab, self.d_model)
        self.layers = [_PerformerLayer(self.d_model, self.nhead, cfg["dim_feedforward"])
                       for _ in range(self.n_layers)]
        self.ln_out = nn.LayerNorm(self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.vocab, bias=False)
        self.pe = _sinusoidal_pe(self.max_seq_len, self.d_model)

    def load_weights(self, w: dict):
        emap = {k: (v.astype(self.dtype) if v.dtype != self.dtype else v)
                for k, v in w.items()}
        self.tok_embed.load_weights([("weight", emap["tok_embed.weight"])], strict=True)
        self.ln_out.load_weights([("weight", emap["ln_out.weight"]),
                                  ("bias", emap["ln_out.bias"])], strict=True)
        self.out_proj.load_weights([("weight", emap["out_proj.weight"])], strict=True)
        for i, lyr in enumerate(self.layers):
            b = f"encoder.layers.{i}."
            lyr.q_proj.load_weights([("weight", emap[b + "q_proj.weight"]),
                                     ("bias", emap[b + "q_proj.bias"])], strict=True)
            lyr.k_proj.load_weights([("weight", emap[b + "k_proj.weight"]),
                                     ("bias", emap[b + "k_proj.bias"])], strict=True)
            lyr.v_proj.load_weights([("weight", emap[b + "v_proj.weight"]),
                                     ("bias", emap[b + "v_proj.bias"])], strict=True)
            lyr.out_proj.load_weights([("weight", emap[b + "out_proj.weight"]),
                                       ("bias", emap[b + "out_proj.bias"])], strict=True)
            lyr.linear1.load_weights([("weight", emap[b + "linear1.weight"]),
                                      ("bias", emap[b + "linear1.bias"])], strict=True)
            lyr.linear2.load_weights([("weight", emap[b + "linear2.weight"]),
                                      ("bias", emap[b + "linear2.bias"])], strict=True)
            lyr.norm1.load_weights([("weight", emap[b + "norm1.weight"]),
                                    ("bias", emap[b + "norm1.bias"])], strict=True)
            lyr.norm2.load_weights([("weight", emap[b + "norm2.weight"]),
                                    ("bias", emap[b + "norm2.bias"])], strict=True)
        mx.eval([m.parameters() for m in (self.tok_embed, self.ln_out, self.out_proj, *self.layers)])

    @classmethod
    def load(cls, weights_dir: str, dtype=mx.float32):
        with open(os.path.join(weights_dir, "cadenza_config.json")) as f:
            cfg = json.load(f)["performer"]
        self = cls(cfg, dtype=dtype)
        w = mx.load(os.path.join(weights_dir, "cadenza_performer.safetensors"))
        self.load_weights(w)
        return self

    def fill(self, ids: mx.array) -> mx.array:
        """ids (B, L) int -> logits (B, L, V)."""
        ids = ids.astype(mx.int32)
        B, L = ids.shape
        key_padding_mask = (ids == self.pad_id)
        h = self.tok_embed(ids)
        h = h + self.pe[:L].astype(h.dtype)
        for lyr in self.layers:
            h = lyr(h, key_padding_mask)
        h = self.ln_out(h)
        return self.out_proj(h)
