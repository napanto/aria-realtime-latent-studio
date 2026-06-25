"""Fit calibrated latent-control directions for AriaVAE (torch, offline).

Encodes kong-TEST windows to ``z = mu`` and ridge-regresses ``mu -> attributes``
(velocity mean/std, note density, IOI entropy, pitch mean/std, pedal fraction).
The ridge weight column ``w_k`` for attribute ``k`` is the control direction:
moving ``z' = z + (Δ / ||w_k||²) · w_k`` changes the probe's predicted attribute
``k`` by ``Δ`` (in raw attribute units) — a labelled, calibrated slider.

Saves ``latent_directions.npz`` (W, per-attr/​dim stats, names) for the runtime.
This mirrors the probe that validated B05 (ridge R² ≈ 0.91), so the directions
are exactly the ones the latent was shown to encode.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer-config", required=True)
    ap.add_argument("--midi-dir", required=True, help="raw kong-TEST MIDIs")
    ap.add_argument("--aria-repo", default="/var/home/antonio/ispr/external/aria")
    ap.add_argument("--ispr", default="/var/home/antonio/ispr/ispr_v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--win", type=int, default=1024)
    ap.add_argument("--ridge-lambda", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    sys.path.insert(0, args.aria_repo)
    sys.path.insert(0, args.ispr)
    import torch
    from ariautils.midi import MidiDict
    from src.model.aria_vae import compute_attributes, ATTRIBUTE_NAMES
    from src.aria_vae_generate import load_aria_vae, load_tokenizer

    dev = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    tok = load_tokenizer(args.tokenizer_config, args.aria_repo)
    model, cfg, _ = load_aria_vae(args.checkpoint, dev, return_payload=True)
    model.eval()

    # Gather MIDIs recursively (PiJAMA tree is artist/album/song.midi). Same
    # tokenisation as load_prompts_from_midi: tok.encode(tok.tokenize(MidiDict)).
    paths = sorted(Path(args.midi_dir).rglob("*.mid")) + sorted(Path(args.midi_dir).rglob("*.midi"))
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(paths))
    prompts = []
    for i in order:
        if len(prompts) >= args.n:
            break
        try:
            ids = tok.encode(tok.tokenize(MidiDict.from_midi(str(paths[int(i)]))))
        except Exception:
            continue
        if len(ids) >= 8:
            prompts.append(np.asarray(ids[:args.win], dtype=np.int64))
    print(f"[fit] {len(prompts)} windows tokenised (from {len(paths)} MIDIs)")

    mus, ys = [], []
    with torch.no_grad():
        for ids in prompts:
            t = torch.from_numpy(ids).long().unsqueeze(0).to(dev)
            mu, _ = model.encoder(t, None)                 # (1,128)
            a = compute_attributes(t, tok)                 # (1,7)
            mus.append(mu.float().cpu().numpy()[0])
            ys.append(a.float().cpu().numpy()[0])
    mu = np.stack(mus); Y = np.stack(ys)                   # (N,128), (N,7)
    N = mu.shape[0]
    print(f"[fit] encoded {N} windows; attrs ~ mean {Y.mean(0).round(2)}")

    mu_mean, mu_std = mu.mean(0), mu.std(0) + 1e-6
    attr_mean, attr_std = Y.mean(0), Y.std(0) + 1e-6

    muc = mu - mu_mean
    Yc = Y - attr_mean
    lam = args.ridge_lambda
    # W = (muc^T muc + lam I)^-1 muc^T Yc      (128 x 7)
    A = muc.T @ muc + lam * np.eye(mu.shape[1])
    W = np.linalg.solve(A, muc.T @ Yc)

    # train/test R² (80/20) sanity — should land near B05's ~0.91 mean
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(N); ntr = int(0.8 * N)
    tr, te = idx[:ntr], idx[ntr:]
    Wtr = np.linalg.solve(muc[tr].T @ muc[tr] + lam * np.eye(mu.shape[1]), muc[tr].T @ Yc[tr])
    pred = muc[te] @ Wtr
    r2 = []
    for k in range(Y.shape[1]):
        yt = Yc[te, k]; ss_tot = float((yt - yt.mean()) ** 2 @ np.ones_like(yt))
        ss_res = float(((yt - pred[:, k]) ** 2).sum())
        r2.append(1 - ss_res / ss_tot if ss_tot > 1e-9 else float("nan"))
    for k, name in enumerate(ATTRIBUTE_NAMES):
        print(f"[fit]   R²[{name:14s}] = {r2[k]:+.3f}   ||w|| = {np.linalg.norm(W[:, k]):.3f}")
    print(f"[fit] mean R² = {np.nanmean(r2):.3f}")

    np.savez(args.out, W=W.astype(np.float32),
             attr_mean=attr_mean.astype(np.float32), attr_std=attr_std.astype(np.float32),
             mu_mean=mu_mean.astype(np.float32), mu_std=mu_std.astype(np.float32),
             r2=np.array(r2, np.float32), names=np.array(list(ATTRIBUTE_NAMES)),
             n=np.int32(N), ridge_lambda=np.float32(lam))
    print(f"[fit] saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
