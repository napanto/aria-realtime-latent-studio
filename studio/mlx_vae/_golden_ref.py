"""Produce a torch golden reference for AriaVAE→MLX parity testing.

Runs with torch (mlbox). Loads the torch AriaVAE, runs it on a FIXED input, and
dumps the intermediate tensors the MLX port must reproduce:

  ids, mu, z(=mu), prefix(8 soft tokens), per-layer z-residuals(16), attrs(7),
  and teacher-forced decode logits over the real-token positions.

The MLX parity script (runs on host) recomputes these from the converted weights
and compares — catching any transpose / key / RoPE / dtype bug in the port.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ispr_src", default="/var/home/antonio/ispr/ispr_v2/src")
    ap.add_argument("--aria_pkg", default="/var/home/antonio/ispr/external/aria")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    sys.path.insert(0, args.aria_pkg)
    sys.path.insert(0, args.ispr_src)
    from model.aria_vae import AriaVAE, AriaVAEConfig  # noqa: E402

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_dict = ck["aria_vae_config"]
    field_names = {f.name for f in dataclasses.fields(AriaVAEConfig)}
    cfg = AriaVAEConfig(**{k: v for k, v in cfg_dict.items() if k in field_names})
    model = AriaVAE(cfg)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    # Only the decoder alias (_dec.*) should be "unexpected"; nothing should be missing.
    real_missing = [k for k in missing if not k.startswith("_dec.")]
    if real_missing:
        raise SystemExit(f"missing weights: {real_missing[:6]}")
    model.eval()
    torch.manual_seed(args.seed)

    V, S = cfg.vocab_size, args.seq_len
    rng = np.random.RandomState(args.seed)
    ids = rng.randint(4, V, size=(1, S)).astype(np.int64)  # avoid specials 0..3
    ids_t = torch.from_numpy(ids)

    with torch.no_grad():
        mu, logvar = model.encoder(ids_t, None)         # (1,128)
        z = mu
        prefix = model.injector(z)                      # (1,8,1536)
        residuals = torch.stack([model.z_adapters[li](z) for li in range(cfg.dec_n_layers)], 0)  # (16,1,1536)
        attrs = model.attr_head(z)                      # (1,7)
        logits = model.decode(ids_t, prefix, z=z)       # (1,S,V)  fp32 decoder

        # bf16-decoder reference: mirrors the MLX runtime (latent fp32 -> z fp32;
        # prefix/residuals/decoder bf16). Validates the decode INTEGRATION (prefix
        # prefill + per-layer residual threading) at the dtype MLX actually uses.
        m16 = AriaVAE(cfg)
        m16.load_state_dict(ck["model"], strict=False)
        m16.eval()
        m16.decoder.to(torch.bfloat16)
        for a in m16.z_adapters:
            a.to(torch.bfloat16)
        prefix16 = prefix.to(torch.bfloat16)
        logits16 = m16.decode(ids_t, prefix16, z=z.to(torch.bfloat16))

    keep = min(8, S)
    np.savez(
        args.out,
        ids=ids.astype(np.int32),
        mu=mu.float().numpy(),
        prefix=prefix.float().numpy(),
        residuals=residuals.float().numpy(),   # (16,1,1536)
        attrs=attrs.float().numpy(),
        logits_tail=logits[:, -keep:, :].float().numpy(),
        logits_tail_bf16=logits16[:, -keep:, :].float().numpy(),
        keep=np.int32(keep),
        cfg=json.dumps({k: cfg_dict[k] for k in sorted(cfg_dict)}),
    )
    print(f"[golden] ids {ids.shape}  mu {tuple(mu.shape)}  prefix {tuple(prefix.shape)} "
          f"resid {tuple(residuals.shape)}  logits_tail {tuple(logits[:, -keep:, :].shape)}")
    print(f"[golden] mu[:6] = {mu[0, :6].tolist()}")
    print(f"[golden] attrs = {attrs[0].tolist()}")
    print(f"[golden] logits_tail argmax (last pos) = {int(logits[0, -1].argmax())}")
    print(f"[golden] saved -> {args.out}")


if __name__ == "__main__":
    main()
