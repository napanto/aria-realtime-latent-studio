"""Fit calibrated latent-control directions for the Cadenza Composer.

Mirrors ``fit_latent_directions.py`` (the AriaVAE probe), adapted to Cadenza:
encodes PiJAMA windows to ``z = μ`` (Composer encoder) and ridge-regresses
``μ → the 7 attributes`` (velocity mean/std, note density, IOI entropy, pitch
mean/std, pedal fraction — the same set as ``eval_cadenza_stagec``). The ridge
weight column ``w_k`` is the control direction: moving ``z' = z + (Δ/||w_k||²)·w_k``
shifts the probe's predicted attribute ``k`` by ``Δ`` raw units — a labelled,
calibrated slider for the Composer's latent.

Runs in torch (mlbox): the MLX Composer and the torch Composer are parity-
verified (encode μ max|Δ| ≈ 5e-6), so the directions fitted in torch apply
verbatim to the MLX runtime. The Composer was trained on the *performance*
PerTok-p cache, so each encoded window decodes to a full performance MIDI from
which the 7 attributes are computed directly.

Saves ``latent_directions_cadenza.npz`` (W, per-attr/dim stats, names, R²).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--composer-ckpt", required=True)
    ap.add_argument("--midi-dir", required=True, help="raw PiJAMA MIDIs (recursive)")
    ap.add_argument("--ispr", default="/var/home/antonio/ispr/ispr_v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--win", type=int, default=384)
    ap.add_argument("--ridge-lambda", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    sys.path.insert(0, args.ispr)
    import torch
    from src.model.cadenza import Cadenza, CadenzaConfig
    from src.data.pertok_tokenizer import PerTokWrapper
    from src.eval_cadenza_stagec import ATTRIBUTE_NAMES, attributes_from_pretty_midi

    dev = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # Composer (performance-cache trained → emits a full performance MIDI).
    cck = torch.load(args.composer_ckpt, map_location=dev, weights_only=False)
    ccfg_d = dict(cck["config"])
    ccfg = CadenzaConfig(**{k: v for k, v in ccfg_d.items()
                            if k in CadenzaConfig.__dataclass_fields__})
    model = Cadenza(ccfg).to(dev)
    model.load_state_dict(cck.get("model_state") or cck.get("model"), strict=True)
    model.eval()
    latent_dim = ccfg.latent_dim
    max_seq = ccfg.max_seq_len

    tok = PerTokWrapper.from_default(cache_root=None, mode="performance")
    pad_id = int(tok.pad_id)

    paths = sorted(Path(args.midi_dir).rglob("*.mid")) + sorted(Path(args.midi_dir).rglob("*.midi"))
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(paths))
    print(f"[fit] {len(paths)} candidate MIDIs; encoding up to {args.n} windows ...")

    mus, ys = [], []
    win = min(args.win, max_seq)
    for i in order:
        if len(mus) >= args.n:
            break
        p = paths[int(i)]
        try:
            ids = tok.encode_midi(str(p))
        except Exception:
            continue
        ids = np.asarray(ids[:win], dtype=np.int64)
        if ids.size < 16:
            continue
        # The attribute target is computed on the SAME window's decoded MIDI
        # (round-trip), so μ and attrs describe the same musical content.
        try:
            pm = tok.decode(ids)
            attrs = attributes_from_pretty_midi(pm)
        except Exception:
            continue
        if attrs is None:
            continue
        padded = np.full(win, pad_id, dtype=np.int64)
        padded[:ids.size] = ids
        t = torch.from_numpy(padded).unsqueeze(0).to(dev)
        attn = (t != pad_id).long()
        with torch.no_grad():
            mu = model.encode(t, attention_mask=attn, sample=False)   # (1, latent)
        mus.append(mu.float().cpu().numpy()[0])
        ys.append(attrs)
    mu = np.stack(mus); Y = np.stack(ys)
    N = mu.shape[0]
    print(f"[fit] encoded {N} windows; attr means ~ {Y.mean(0).round(2)}")

    mu_mean, mu_std = mu.mean(0), mu.std(0) + 1e-6
    attr_mean, attr_std = Y.mean(0), Y.std(0) + 1e-6
    muc = mu - mu_mean
    Yc = Y - attr_mean
    lam = args.ridge_lambda
    A = muc.T @ muc + lam * np.eye(latent_dim)
    W = np.linalg.solve(A, muc.T @ Yc)                                # (latent, n_attr)

    # held-out R² (80/20).
    rng2 = np.random.default_rng(args.seed)
    idx = rng2.permutation(N); ntr = int(0.8 * N)
    tr, te = idx[:ntr], idx[ntr:]
    Wtr = np.linalg.solve(muc[tr].T @ muc[tr] + lam * np.eye(latent_dim), muc[tr].T @ Yc[tr])
    pred = muc[te] @ Wtr
    r2 = []
    for k in range(Y.shape[1]):
        yt = Yc[te, k]
        ss_tot = float(((yt - yt.mean()) ** 2).sum())
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
