"""Cadenza two-stage generation in MLX → MIDI. Runs on the host (mlx + miditok).

Full pipeline (mirrors torch ``src/cadenza_two_stage_generate.py``, adapted to
THIS checkpoint where the Composer was trained on the *performance* PerTok-p
cache and therefore emits velocity/microtime/pedal tokens directly):

  seed MIDI ──encode──▶ z (Composer μ)  [optionally manipulate z]
            ──generate──▶ performance tokens (Composer, KV-cached AR)
            ──mask──▶ blank every Velocity / MicroTime / Pedal slot with MASK
            ──fill──▶ Performer re-predicts those slots (bidirectional)
            ──detok──▶ PerTok-p → MIDI

Two stage-2 modes are written per seed:
  * ``composer_only``  — the Composer's raw emission, detokenised directly.
  * ``two_stage``      — masked then Performer-refilled (the paper's pipeline).

This is the automatable validation of the whole chain: it must write VALID
.mid files. Latency for each phase is reported.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from cadenza_mlx import CadenzaComposerMLX, CadenzaPerformerMLX


PERFORMANCE_PREFIXES = ("Velocity_", "MicroTiming_", "Pedal_", "PedalOff_")


def load_perf_tok(ispr_path: str):
    """Build the PerTok-p wrapper (mode=performance) used to (de)tokenise."""
    sys.path.insert(0, ispr_path)
    from src.data.pertok_tokenizer import PerTokWrapper
    return PerTokWrapper.from_default(cache_root=None, mode="performance")


def encode_seed(perf_tok, midi_path: str, max_len: int) -> np.ndarray:
    """Tokenise a seed MIDI into compact PerTok-p ids (truncate to max_len)."""
    ids = perf_tok.encode_midi(midi_path)
    return np.asarray(ids[:max_len], dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_dir", required=True)
    ap.add_argument("--seed_midi", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ispr", default=None,
                    help="path to ispr_v2 (for the PerTok wrapper). If absent, "
                         "vendored copy under the studio is used.")
    ap.add_argument("--prompt_len", type=int, default=384)
    ap.add_argument("--max_generate_steps", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--performer_sample", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ispr = args.ispr or str(Path(__file__).resolve())
    perf_tok = load_perf_tok(args.ispr) if args.ispr else load_perf_tok(
        str(Path(__file__).resolve().parents[2]))
    perf_id_set = set(perf_tok.token_ids_by_prefix(PERFORMANCE_PREFIXES))
    mask_id = int(perf_tok.mask_id)
    pad_id = int(perf_tok.pad_id)
    eos_id = int(perf_tok.eos_id)
    print(f"[tok] perf vocab={perf_tok.vocab_size}  n_perf_ids={len(perf_id_set)}  mask_id={mask_id}")

    composer = CadenzaComposerMLX.load(args.weights_dir, dtype=mx.float32)
    performer = CadenzaPerformerMLX.load(args.weights_dir, dtype=mx.float32)

    # --- seed → z ---
    prompt = encode_seed(perf_tok, args.seed_midi, min(args.prompt_len, composer.cfg["max_seq_len"]))
    if prompt.size < 8:
        raise SystemExit(f"seed produced only {prompt.size} tokens")
    t0 = time.time()
    mu = composer.encode(mx.array(prompt[None, :]))
    mx.eval(mu)
    enc_ms = (time.time() - t0) * 1000
    z = mu
    print(f"[gen] seed '{Path(args.seed_midi).name}' -> {prompt.size} prompt toks; encode {enc_ms:.0f} ms")

    # --- z → composer tokens (AR) ---
    t0 = time.time()
    gen = composer.generate(z, max_steps=args.max_generate_steps,
                            temperature=args.temperature, top_k=args.top_k,
                            top_p=args.top_p, key=mx.random.key(args.seed))
    mx.eval(gen)
    gen_ids = np.array(gen[0]).astype(np.int32)
    gen_ms = (time.time() - t0) * 1000
    # crop on EOS / strip trailing PAD
    eos_pos = np.where(gen_ids == eos_id)[0]
    if eos_pos.size:
        gen_ids = gen_ids[: int(eos_pos[0])]
    nonpad = np.where(gen_ids != pad_id)[0]
    if nonpad.size:
        gen_ids = gen_ids[: int(nonpad[-1]) + 1]
    n_gen = gen_ids.size
    tps = n_gen / (gen_ms / 1000) if gen_ms > 0 else 0
    print(f"[gen] composer emitted {n_gen} tokens in {gen_ms:.0f} ms ({tps:.0f} tok/s)")

    if n_gen < 4:
        raise SystemExit("composer emitted too few tokens")

    # --- composer-only MIDI (sanity) ---
    seed_name = Path(args.seed_midi).stem
    comp_path = out_dir / f"{seed_name}_composer_only.mid"
    info = {"seed": seed_name, "n_prompt": int(prompt.size), "n_gen": int(n_gen)}
    try:
        comp_midi = perf_tok.decode(gen_ids)
        comp_midi.write(str(comp_path))
        n_notes = sum(len(i.notes) for i in comp_midi.instruments)
        info["composer_only_ok"] = True
        info["composer_only_notes"] = n_notes
        print(f"[gen] composer_only -> {comp_path.name}  ({n_notes} notes)")
    except Exception as e:
        info["composer_only_ok"] = False
        info["composer_only_error"] = f"{type(e).__name__}: {e}"
        print(f"[gen] composer_only FAILED: {e}")

    # --- two-stage: mask expressive slots, Performer fill ---
    perf_in = gen_ids.copy()
    mask_positions = np.array([tid in perf_id_set for tid in gen_ids], dtype=bool)
    perf_in[mask_positions] = mask_id
    # truncate to performer max_seq_len
    plen = min(perf_in.size, performer.max_seq_len)
    perf_in = perf_in[:plen]
    mask_positions = mask_positions[:plen]
    n_masks = int(mask_positions.sum())

    t0 = time.time()
    plogits = performer.fill(mx.array(perf_in[None, :]))
    mx.eval(plogits)
    fill_ms = (time.time() - t0) * 1000
    plogits_np = np.array(plogits[0])
    out_ids = perf_in.copy()
    midx = np.where(mask_positions)[0]
    if args.performer_sample:
        for p in midx:
            row = plogits_np[p] / max(args.temperature, 1e-9)
            row = row - row.max()
            probs = np.exp(row); probs /= probs.sum()
            out_ids[p] = int(np.random.choice(len(probs), p=probs))
    else:
        out_ids[midx] = plogits_np[midx].argmax(-1).astype(np.int32)
    print(f"[gen] performer filled {n_masks} masked slots in {fill_ms:.0f} ms")

    ts_path = out_dir / f"{seed_name}_two_stage.mid"
    try:
        ts_midi = perf_tok.decode(out_ids)
        ts_midi.write(str(ts_path))
        n_notes = sum(len(i.notes) for i in ts_midi.instruments)
        info["two_stage_ok"] = True
        info["two_stage_notes"] = n_notes
        info["n_masks"] = n_masks
        print(f"[gen] two_stage -> {ts_path.name}  ({n_notes} notes)")
    except Exception as e:
        info["two_stage_ok"] = False
        info["two_stage_error"] = f"{type(e).__name__}: {e}"
        print(f"[gen] two_stage FAILED: {e}")

    info["latency_ms"] = {"encode": enc_ms, "generate": gen_ms, "fill": fill_ms,
                          "tok_per_s": tps}
    import json
    (out_dir / f"{seed_name}_info.json").write_text(json.dumps(info, indent=2))
    print(f"[done] {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
