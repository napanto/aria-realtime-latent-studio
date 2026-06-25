"""Convert torch Cadenza checkpoints (Composer + Performer ``best.pt``) into
MLX-loadable artifacts. Mirrors ``convert.py`` (the AriaVAE converter).

Runs with **torch** (in ``mlbox``); no MLX needed here. Produces, under
``<out_dir>``:

  * ``cadenza_composer.safetensors`` — the Composer VAE (encoder + W_pre +
    AdaLN gates + decoder + tied lm_head), keyed exactly as ``cadenza_mlx``'s
    ``CadenzaComposerMLX`` expects (same names as the torch ``Cadenza``
    module: ``tok_embed.weight``, ``enc_blocks.{i}.*``, ``W_pre.weight``,
    ``dec_blocks.{i}.*`` incl. ``adaln.1.{weight,bias}``, ``mu_head.*``,
    ``logvar_head.*``, ``lm_head.weight``). Saved **fp32** (parity-critical,
    ~75 MB).
  * ``cadenza_performer.safetensors`` — the Performer, with PyTorch's fused
    ``nn.TransformerEncoderLayer`` weights SPLIT into the per-head q/k/v form
    MLX needs (``in_proj_weight`` → ``{q,k,v}_proj.{weight,bias}``). Saved
    fp32 (~85 MB; quantised on load if requested).
  * ``cadenza_config.json`` — both model configs + tokenizer ids.

Unlike AriaVAE (which bolts a latent onto a frozen *external* aria decoder),
the Cadenza Composer is self-contained (its own decoder), and the Composer
here was trained on the **performance** PerTok cache (vocab 220) — i.e. it
emits velocity/microtime/pedal tokens directly. The Performer then re-fills
the masked expressive slots. Both vocabs are 220, so id spaces match.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from safetensors.torch import save_file


def _composer_state(sd: dict) -> dict:
    """Pass the Composer state through verbatim (names already match the MLX
    port). Drop the duplicated tied lm_head if it aliases tok_embed — we keep
    lm_head.weight explicitly so the MLX side can load it without relying on
    Python-level tying."""
    out = {}
    for k, v in sd.items():
        out[k] = v.float().contiguous()
    # lm_head.weight is tied to tok_embed.weight (shared storage). Clone it so
    # safetensors doesn't reject the duplicate-memory aliasing; the MLX side
    # loads both keys independently.
    if "lm_head.weight" in out and "tok_embed.weight" in out:
        out["lm_head.weight"] = out["lm_head.weight"].clone().contiguous()
    return out


def _performer_state(sd: dict, n_layers: int) -> dict:
    """Translate PyTorch ``nn.TransformerEncoder`` weights into MLX-friendly
    names. PyTorch fuses q/k/v into one ``in_proj_weight`` (3D, D); we split
    it into three (D, D) blocks. Everything else maps by a simple rename.

    PyTorch layer param names (per layer i):
        encoder.layers.{i}.self_attn.in_proj_weight   (3D, D)
        encoder.layers.{i}.self_attn.in_proj_bias     (3D,)
        encoder.layers.{i}.self_attn.out_proj.weight  (D, D)
        encoder.layers.{i}.self_attn.out_proj.bias    (D,)
        encoder.layers.{i}.linear1.weight             (ff, D)   + bias
        encoder.layers.{i}.linear2.weight             (D, ff)   + bias
        encoder.layers.{i}.norm1.{weight,bias}        (D,)
        encoder.layers.{i}.norm2.{weight,bias}        (D,)
    plus tok_embed.weight, ln_out.{weight,bias}, out_proj.weight (tied).
    """
    out = {}
    # Embedding + final LN + tied out_proj.
    for k in ("tok_embed.weight", "ln_out.weight", "ln_out.bias", "out_proj.weight"):
        if k in sd:
            out[k] = sd[k].float().contiguous()
    # out_proj is tied to tok_embed; ensure it exists and is NOT a storage
    # alias (safetensors rejects duplicate memory).
    if "out_proj.weight" not in out and "tok_embed.weight" in out:
        out["out_proj.weight"] = out["tok_embed.weight"].clone().contiguous()
    elif "out_proj.weight" in out and "tok_embed.weight" in out:
        out["out_proj.weight"] = out["out_proj.weight"].clone().contiguous()

    for i in range(n_layers):
        b = f"encoder.layers.{i}."
        ipw = sd[b + "self_attn.in_proj_weight"].float()   # (3D, D)
        ipb = sd[b + "self_attn.in_proj_bias"].float()     # (3D,)
        D = ipw.shape[1]
        qw, kw, vw = ipw[:D], ipw[D:2 * D], ipw[2 * D:]
        qb, kb, vb = ipb[:D], ipb[D:2 * D], ipb[2 * D:]
        out[b + "q_proj.weight"] = qw.contiguous()
        out[b + "k_proj.weight"] = kw.contiguous()
        out[b + "v_proj.weight"] = vw.contiguous()
        out[b + "q_proj.bias"] = qb.contiguous()
        out[b + "k_proj.bias"] = kb.contiguous()
        out[b + "v_proj.bias"] = vb.contiguous()
        out[b + "out_proj.weight"] = sd[b + "self_attn.out_proj.weight"].float().contiguous()
        out[b + "out_proj.bias"] = sd[b + "self_attn.out_proj.bias"].float().contiguous()
        out[b + "linear1.weight"] = sd[b + "linear1.weight"].float().contiguous()
        out[b + "linear1.bias"] = sd[b + "linear1.bias"].float().contiguous()
        out[b + "linear2.weight"] = sd[b + "linear2.weight"].float().contiguous()
        out[b + "linear2.bias"] = sd[b + "linear2.bias"].float().contiguous()
        out[b + "norm1.weight"] = sd[b + "norm1.weight"].float().contiguous()
        out[b + "norm1.bias"] = sd[b + "norm1.bias"].float().contiguous()
        out[b + "norm2.weight"] = sd[b + "norm2.weight"].float().contiguous()
        out[b + "norm2.bias"] = sd[b + "norm2.bias"].float().contiguous()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--composer-ckpt", required=True, help="Composer best.pt")
    ap.add_argument("--performer-ckpt", required=True, help="Performer best.pt")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Composer -------------------------------------------------------
    print(f"[convert] loading composer {args.composer_ckpt} ...")
    cck = torch.load(args.composer_ckpt, map_location="cpu", weights_only=False)
    csd = cck.get("model_state") or cck.get("model") or cck
    ccfg = dict(cck["config"])
    composer = _composer_state(csd)

    n_dec = ccfg["n_layers_dec"]
    n_enc = ccfg["n_layers_enc"]
    need = ["tok_embed.weight", "W_pre.weight", "mu_head.weight", "logvar_head.weight",
            "enc_ln_out.weight", "dec_ln_out.weight", "lm_head.weight"]
    for i in range(n_enc):
        need += [f"enc_blocks.{i}.attn.qkv.weight", f"enc_blocks.{i}.attn.out.weight",
                 f"enc_blocks.{i}.ff.0.weight", f"enc_blocks.{i}.ff.3.weight",
                 f"enc_blocks.{i}.ln1.weight", f"enc_blocks.{i}.ln2.weight"]
    for i in range(n_dec):
        need += [f"dec_blocks.{i}.attn.qkv.weight", f"dec_blocks.{i}.attn.out.weight",
                 f"dec_blocks.{i}.ff.0.weight", f"dec_blocks.{i}.ff.3.weight",
                 f"dec_blocks.{i}.ln1.weight", f"dec_blocks.{i}.ln2.weight"]
        if ccfg.get("z_adaln"):
            need += [f"dec_blocks.{i}.adaln.1.weight", f"dec_blocks.{i}.adaln.1.bias"]
    missing = [k for k in need if k not in composer]
    if missing:
        raise SystemExit(f"[convert] FATAL: composer missing {len(missing)} keys, "
                         f"e.g. {missing[:5]}")

    # --- Performer ------------------------------------------------------
    print(f"[convert] loading performer {args.performer_ckpt} ...")
    pck = torch.load(args.performer_ckpt, map_location="cpu", weights_only=False)
    psd = pck.get("model_state") or pck.get("model") or pck
    pcfg = dict(pck["config"])
    performer = _performer_state(psd, pcfg["n_layers"])

    pneed = ["tok_embed.weight", "ln_out.weight", "ln_out.bias", "out_proj.weight"]
    for i in range(pcfg["n_layers"]):
        b = f"encoder.layers.{i}."
        pneed += [b + s for s in ("q_proj.weight", "k_proj.weight", "v_proj.weight",
                                  "out_proj.weight", "linear1.weight", "linear2.weight",
                                  "norm1.weight", "norm2.weight")]
    pmissing = [k for k in pneed if k not in performer]
    if pmissing:
        raise SystemExit(f"[convert] FATAL: performer missing {pmissing[:5]}")

    cpath = os.path.join(args.out_dir, "cadenza_composer.safetensors")
    ppath = os.path.join(args.out_dir, "cadenza_performer.safetensors")
    save_file(composer, cpath, metadata={"format": "pt"})
    save_file(performer, ppath, metadata={"format": "pt"})

    config = {
        "composer": {
            "vocab_size": ccfg["vocab_size"], "pad_id": ccfg["pad_id"],
            "bos_id": ccfg["bos_id"], "eos_id": ccfg["eos_id"], "mask_id": ccfg["mask_id"],
            "max_seq_len": ccfg["max_seq_len"], "d_model": ccfg["d_model"],
            "n_layers_enc": ccfg["n_layers_enc"], "n_heads_enc": ccfg["n_heads_enc"],
            "d_ff_enc": ccfg["d_ff_enc"], "n_layers_dec": ccfg["n_layers_dec"],
            "n_heads_dec": ccfg["n_heads_dec"], "d_ff_dec": ccfg["d_ff_dec"],
            "latent_dim": ccfg["latent_dim"], "z_adaln": bool(ccfg.get("z_adaln", False)),
            "rope_base": 10000.0,
        },
        "performer": {
            "vocab_size": pcfg["vocab_size"], "pad_id": pcfg["pad_id"],
            "bos_id": pcfg["bos_id"], "eos_id": pcfg["eos_id"], "mask_id": pcfg["mask_id"],
            "max_seq_len": pcfg["max_seq_len"], "d_model": pcfg["d_model"],
            "nhead": pcfg["nhead"], "n_layers": pcfg["n_layers"],
            "dim_feedforward": pcfg["dim_feedforward"],
        },
        "composer_step": cck.get("step"),
        "performer_step": pck.get("step"),
        "composer_cache_root": cck.get("cache_root"),
    }
    with open(os.path.join(args.out_dir, "cadenza_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    csz = os.path.getsize(cpath) / 1e6
    psz = os.path.getsize(ppath) / 1e6
    print(f"[convert] composer : {len(composer)} tensors -> {cpath} ({csz:.0f} MB, fp32)")
    print(f"[convert] performer: {len(performer)} tensors -> {ppath} ({psz:.0f} MB, fp32)")
    print(f"[convert] z_adaln={config['composer']['z_adaln']}  "
          f"composer vocab={config['composer']['vocab_size']}  "
          f"performer vocab={config['performer']['vocab_size']}")
    print("[convert] OK")


if __name__ == "__main__":
    main()
