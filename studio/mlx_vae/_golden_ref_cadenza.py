"""Torch golden reference for the Cadenza MLX port. Runs in mlbox (torch).

Loads the torch Composer + Performer, runs them on FIXED inputs, and dumps the
intermediates the MLX port must reproduce:

  Composer: ids, mu (encode pos-0 pool), z(=mu), teacher-forced decode logits
            over the BOS-shifted input (fp32) + a bf16-decoder reference.
  Performer: perf_ids (with MASK at random positions), logits (fp32) + bf16 ref.

The MLX parity script recomputes these from the converted safetensors and
compares — catching any transpose / RoPE / AdaLN / id-split bug.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--composer-ckpt", required=True)
    ap.add_argument("--performer-ckpt", required=True)
    ap.add_argument("--ispr", default="/var/home/antonio/ispr/ispr_v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=192)
    ap.add_argument("--perf-seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    sys.path.insert(0, args.ispr)
    from src.model.cadenza import Cadenza, CadenzaConfig
    from src.model.cadenza_performer import CadenzaPerformer, PerformerConfig

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    # --- Composer ------------------------------------------------------
    cck = torch.load(args.composer_ckpt, map_location="cpu", weights_only=False)
    ccfg_d = dict(cck["config"])
    ccfg = CadenzaConfig(**{k: v for k, v in ccfg_d.items()
                            if k in CadenzaConfig.__dataclass_fields__})
    composer = Cadenza(ccfg)
    composer.load_state_dict(cck.get("model_state") or cck.get("model"), strict=True)
    composer.eval()

    V = ccfg.vocab_size
    S = args.seq_len
    ids = rng.randint(4, V, size=(1, S)).astype(np.int64)   # avoid specials
    ids_t = torch.from_numpy(ids)
    # decoder input: BOS prepended, shift right (matches forward()).
    dec_in = np.concatenate([[[ccfg.bos_id]], ids[:, :-1]], axis=1).astype(np.int64)
    dec_in_t = torch.from_numpy(dec_in)

    with torch.no_grad():
        mu, logvar, _ = composer._encode(ids_t, None)       # (1, latent)
        z = mu
        logits, _ = composer._decode(z, dec_in_t, kv_caches=None, return_cache=False)  # (1, S, V)

        # bf16-decoder reference (the dtype a quantised host might use): keep
        # the fp32 encoder/z, run only the decoder + heads in bf16 so the
        # parity check's bf16 comparison reflects the host's runtime regime.
        comp16 = Cadenza(ccfg)
        comp16.load_state_dict(cck.get("model_state") or cck.get("model"), strict=True)
        comp16.eval()
        comp16.W_pre.to(torch.bfloat16)
        comp16.dec_blocks.to(torch.bfloat16)
        comp16.dec_ln_out.to(torch.bfloat16)
        comp16.lm_head.to(torch.bfloat16)
        comp16.tok_embed.to(torch.bfloat16)
        logits16, _ = comp16._decode(z.to(torch.bfloat16), dec_in_t,
                                     kv_caches=None, return_cache=False)

    # --- Performer -----------------------------------------------------
    pck = torch.load(args.performer_ckpt, map_location="cpu", weights_only=False)
    pcfg_d = dict(pck["config"])
    pcfg = PerformerConfig(**{k: v for k, v in pcfg_d.items()
                              if k in PerformerConfig.__dataclass_fields__})
    performer = CadenzaPerformer(pcfg)
    performer.load_state_dict(pck.get("model_state") or pck.get("model"), strict=True)
    performer.eval()

    PV = pcfg.vocab_size
    PS = args.perf_seq_len
    perf_ids = rng.randint(4, PV, size=(1, PS)).astype(np.int64)
    # mask ~1/3 of positions with MASK id.
    mask_pos = (rng.rand(PS) < 0.34)
    perf_in = perf_ids.copy()
    perf_in[0, mask_pos] = pcfg.mask_id
    perf_in_t = torch.from_numpy(perf_in)
    with torch.no_grad():
        plogits = performer(perf_in_t, attention_mask=torch.ones_like(perf_in_t))  # (1, PS, PV)

    keep = min(8, S)
    pkeep = min(8, PS)
    np.savez(
        args.out,
        ids=ids.astype(np.int32),
        dec_in=dec_in.astype(np.int32),
        mu=mu.float().numpy(),
        logits_tail=logits[:, -keep:, :].float().numpy(),
        logits_tail_bf16=logits16[:, -keep:, :].float().numpy(),
        keep=np.int32(keep),
        perf_in=perf_in.astype(np.int32),
        perf_mask=mask_pos.astype(np.bool_),
        plogits_tail=plogits[:, -pkeep:, :].float().numpy(),
        pkeep=np.int32(pkeep),
        ccfg=json.dumps({k: ccfg_d[k] for k in sorted(ccfg_d)}),
        pcfg=json.dumps({k: pcfg_d[k] for k in sorted(pcfg_d)}),
    )
    print(f"[golden] composer ids {ids.shape} mu {tuple(mu.shape)} logits {tuple(logits.shape)}")
    print(f"[golden] mu[:6] = {mu[0, :6].tolist()}")
    print(f"[golden] composer decode argmax (last pos) = {int(logits[0, -1].argmax())}")
    print(f"[golden] performer perf_in {perf_in.shape} ({int(mask_pos.sum())} masked) "
          f"plogits {tuple(plogits.shape)}")
    print(f"[golden] performer fill argmax (first masked) = "
          f"{int(plogits[0, np.where(mask_pos)[0][0]].argmax())}")
    print(f"[golden] saved -> {args.out}")


if __name__ == "__main__":
    main()
