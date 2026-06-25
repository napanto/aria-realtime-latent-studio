"""Convert a torch AriaVAE checkpoint (``last.pt``) into MLX-loadable artifacts.

Runs with **torch** (in ``mlbox``); no MLX needed here. Produces, under
``<out_dir>``:

  * ``aria_vae_decoder.safetensors``  — the frozen Aria decoder, keyed exactly as
    aria's MLX ``TransformerLM`` expects (``model.tok_embeddings.weight``,
    ``model.encode_layers.{i}.*``, ``model.out_layer_norm.*``, ``lm_head.weight``).
    Saved in **bf16** (the demo's runtime dtype; it int8-quantises on load anyway).
  * ``aria_vae_latent.safetensors`` — the trainable add-on (encoder / injector /
    attr_head / z_adapters), saved in **fp32** (tiny, ~50 MB, parity-critical).
  * ``aria_vae_config.json`` — the ``aria_vae_config`` dict + a derived decoder
    ``ModelConfig`` so the studio can instantiate matching shapes.

The decoder is shipped self-contained (extracted from the *same* checkpoint the
latent was trained against), so the studio never has to assume the host's
``aria-real-time-jazz.safetensors`` is byte-identical to B05's frozen decoder.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from safetensors.torch import save_file

# aria's MLX TransformerLM param names (target side). The torch checkpoint stores
# the decoder under "decoder.model.*" / "decoder.lm_head.*"; aria's MLX model
# wants "model.*" / "lm_head.*". The inner block/proj names already match
# (mixed_qkv, att_proj_linear, ff_gate_proj, ff_up_proj, ff_down_proj, norm1,
# norm2, out_layer_norm), so stripping the leading "decoder." is the only remap.
LATENT_PREFIXES = ("encoder.", "injector.", "attr_head.", "z_adapters.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to AriaVAE last.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--decoder_dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[convert] loading {args.ckpt} ...")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"]
    cfg = ck["aria_vae_config"]

    dec_dtype = getattr(torch, args.decoder_dtype)

    decoder, latent = {}, {}
    skipped = 0
    for k, v in sd.items():
        if k.startswith("decoder."):
            # "decoder.model.X" -> "model.X" ; "decoder.lm_head.X" -> "lm_head.X"
            decoder[k[len("decoder."):]] = v.to(dec_dtype).contiguous()
        elif k.startswith("_dec."):
            skipped += 1  # alias of decoder.model.* — drop the duplicate
        elif k.startswith(LATENT_PREFIXES):
            latent[k] = v.float().contiguous()
        else:
            skipped += 1
            print(f"[convert]   (unclassified, skipped) {k} {tuple(v.shape)}")

    # Sanity: the decoder must carry every weight aria's TransformerLM needs.
    n_layers = cfg["dec_n_layers"]
    need = ["model.tok_embeddings.weight", "model.out_layer_norm.weight",
            "model.out_layer_norm.bias", "lm_head.weight"]
    for i in range(n_layers):
        b = f"model.encode_layers.{i}."
        need += [b + s for s in ("mixed_qkv.weight", "att_proj_linear.weight",
                                 "ff_gate_proj.weight", "ff_up_proj.weight",
                                 "ff_down_proj.weight", "norm1.weight",
                                 "norm1.bias", "norm2.weight", "norm2.bias")]
    missing = [k for k in need if k not in decoder]
    if missing:
        raise SystemExit(f"[convert] FATAL: decoder missing {len(missing)} keys, "
                         f"e.g. {missing[:4]}")

    # Latent sanity.
    lat_need = (["encoder.tok_emb.weight", "encoder.norm.weight",
                 "encoder.to_latent.weight", "encoder.to_latent.bias",
                 "injector.proj.weight", "injector.proj.bias"]
                + [f"z_adapters.{i}.weight" for i in range(n_layers)])
    lat_missing = [k for k in lat_need if k not in latent]
    if lat_missing:
        raise SystemExit(f"[convert] FATAL: latent missing {lat_missing}")

    dpath = os.path.join(args.out_dir, "aria_vae_decoder.safetensors")
    lpath = os.path.join(args.out_dir, "aria_vae_latent.safetensors")
    save_file(decoder, dpath, metadata={"format": "pt"})
    save_file(latent, lpath, metadata={"format": "pt"})

    # Derived MLX ModelConfig for the decoder (medium, no emb path — dec_emb_size
    # is None in B05; AriaVAE injects via the prefix, not the contrastive emb).
    model_config = {
        "d_model": cfg["dec_d_model"],
        "n_heads": cfg["dec_n_heads"],
        "n_layers": cfg["dec_n_layers"],
        "ff_mult": cfg["dec_ff_mult"],
        "max_seq_len": cfg["dec_max_seq_len"],
        "vocab_size": cfg["vocab_size"],
        "emb_size": cfg["dec_emb_size"],
        "drop_p": 0.0,
    }
    with open(os.path.join(args.out_dir, "aria_vae_config.json"), "w") as f:
        json.dump({"aria_vae_config": cfg, "decoder_model_config": model_config,
                   "step": ck.get("step"), "git_sha": ck.get("git_sha")}, f, indent=2)

    dsz = os.path.getsize(dpath) / 1e6
    lsz = os.path.getsize(lpath) / 1e6
    print(f"[convert] decoder: {len(decoder)} tensors -> {dpath} ({dsz:.0f} MB, {args.decoder_dtype})")
    print(f"[convert] latent : {len(latent)} tensors -> {lpath} ({lsz:.1f} MB, fp32)")
    print(f"[convert] skipped {skipped} keys (aliases/non-model)")
    print("[convert] OK")


if __name__ == "__main__":
    main()
