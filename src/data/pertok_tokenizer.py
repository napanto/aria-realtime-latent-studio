"""Thin PerTok wrapper for the Cadenza Transformer-VAE.

PerTok (Lemonaide / miditok) is the "Performance Tokenizer" — every note
gets 2-5 tokens (TimeShift, Pitch, [Velocity], [MicroTiming], [Duration])
with explicit microtiming deltas, which is exactly what expressive piano
generation needs (no grid quantization).

Cadenza's two-stage architecture (Lenz et al. 2024, arXiv 2410.02060)
requires TWO different PerTok configurations:

  * **Composition-only PerTok** (paper Table 1 "PerTok": vocab ~196)
    — Pitch, Duration, TimeShift, Bar. NO Velocity, NO MicroTime.
    Used by the **Composer** (§3.1) which generates beat-quantized score
    tokens auto-regressively from the latent.
  * **PerTok-p** (paper Table 1 "PerTok-p": vocab ~259, plus +2 pedal
    tokens here per the user's pedal request → ~261) — adds Velocity
    and MicroTime. Used by the **Performer** (§3.2) which is a BERT-
    style encoder that re-fills `[MASK]` positions placed over every
    Velocity AND MicroTime token. We additionally enable miditok's
    ``use_sustain_pedals`` so sustain on/off events are part of the
    Performer's vocab (one of the few performance attributes left out
    of the original paper).
  * **Legacy** — the pre-refactor behaviour (Velocity but no MicroTime,
    no Pedal). Retained for backwards compatibility with the existing
    ``/dev/shm`` token cache; new training MUST use either
    ``composition`` or ``performance``.

Mode is plumbed end-to-end:

  * ``PerTokWrapper.from_default(mode=...)`` selects which PerTok config to
    build.
  * ``cache_root/meta.json`` records the mode so downstream loaders can
    refuse mismatched checkpoints (Composer ↔ Performer caches must not
    be silently swapped).
  * Each mode targets a different cache directory by convention
    (``..._compo``, ``..._perf``, legacy ``...``).

Settings here follow the Cadenza paper §2.2 / §4.1 / Table 1:
ticks-per-quarter=440, num_velocities=32, default beat_res grid.
``num_microtiming_bins`` is unspecified in the paper so we keep our
existing value of 30 (a ±60 ms range at 120 BPM, fine-grained enough for
human-level microtiming).

This wrapper:

  * builds the miditok ``PerTok`` instance with project-default settings;
  * tokenizes a directory of ``.mid`` / ``.midi`` files in bulk;
  * caches the resulting integer arrays to a single ``.npz``/``.json``
    pair under ``cache_root`` so subsequent epochs / runs reuse them;
  * exposes ``pad_id``, ``bos_id``, ``eos_id``, ``mask_id``,
    ``vocab_size`` attributes so downstream code doesn't have to poke at
    miditok internals;
  * round-trips ``decode(token_ids)`` → ``pretty_midi.PrettyMIDI`` via a
    ``symusic.Score`` → in-memory MIDI bytes → ``pretty_midi`` chain.

Usage
-----
    tok = PerTokWrapper.from_default(
        mode="composition",
        cache_root="data/processed/aria_midi_pertok_compo",
    )
    tok.tokenize_directory(midi_dir, recursive=True, num_workers=4)
    ids: list[np.ndarray] = tok.load_cache()
    midi = tok.decode(ids[0])     # PrettyMIDI for inspection / audio rendering

Smoke test at the bottom (`python -m src.data.pertok_tokenizer`) runs the
tokenize → cache → load → decode round-trip for all three modes on 5
PiJAMA MIDI files.
"""
from __future__ import annotations

import io
import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# PerTok modes — one dataclass per Cadenza-paper variant.
# ---------------------------------------------------------------------------


PerTokMode = Literal["composition", "performance", "legacy_velocity"]


# ---------------------------------------------------------------------------
# NOTE on observed vocab sizes (Audit checklist anchor)
#
# Paper Table 1 reports PerTok vocab = 196 and PerTok-p vocab = 259, but
# the table does NOT list every knob the authors used. With our defaults
# (pitch_range=(21,109), num_velocities=32, num_microtiming_bins=30,
# beat_res={(0,4):4,(4,12):2}, no chords/rests/tempos/programs/timesig/
# pitchdrum) we observe:
#
#   composition : vocab = 156
#   performance : vocab = 221  (incl. 2 pedal tokens, our user-requested
#                                extension beyond the paper's PerTok-p)
#
# The gap (~40 vocab ids) is consistent with the paper using either a
# wider pitch_range, more duration bins (more bars covered in beat_res),
# or a slightly larger num_microtiming_bins. Architecture and method
# (composition-only vs. PerTok-p split, MASK-100% Performer training)
# are paper-faithful; the exact vocab size is a knob-tuning question the
# paper underspecifies.
# ---------------------------------------------------------------------------


@dataclass
class PerTokSettings:
    """Configuration for a single PerTok variant.

    Defaults below match the **composition-only** variant (paper Table 1
    "PerTok": vocab ~196). The factory helpers ``settings_for_mode`` below
    flip the relevant fields for performance / legacy modes so the only
    place that knows the per-variant deltas is this file.
    """

    pitch_range: tuple[int, int] = (21, 109)            # 88-key piano range
    ticks_per_quarter: int = 440                        # Cadenza paper §4.1 / Table 1
    beat_res: tuple[tuple[tuple[int, int], int], ...] = (
        ((0, 4), 4),                                    # quarters → 16th grid
        ((4, 12), 2),                                   # bars 4-12 → 8th grid
    )
    # Composition-only defaults: no microtime, no velocity, no pedal.
    use_microtiming: bool = False
    max_microtiming_shift: float = 0.125                # ±1/8 beat, ~ ±60 ms @ 120 BPM
    num_microtiming_bins: int = 30
    num_velocities: int = 32                            # paper §4.1: 32 velocity buckets
    # Paper §2.2: PerTok aims at single-track piano with a fixed tempo;
    # the paper notes "music producers often keep a singular, consistent
    # tempo throughout their composition" and explicitly avoids tempo
    # tokens. Time signatures are not mentioned in the paper's PerTok
    # vocab breakdown (Table 1 vocab 196 / 259 is reproducible only
    # with use_time_signatures=False), so we default to False here.
    # 2026-06-03 audit ratification: keep use_time_signatures=False; it
    # holds the vocab closer to paper Table 1.
    use_time_signatures: bool = False
    use_velocities: bool = False
    use_durations: bool = True
    use_sustain_pedals: bool = False                    # performance-only by default
    sustain_pedal_duration: bool = False
    # We're solo piano — drum-pitch tokens just inflate the vocab with
    # ids we will never emit. 2026-06-03 audit ratification: keep
    # use_pitchdrum_tokens=False (holds vocab closer to paper Table 1).
    use_pitchdrum_tokens: bool = False
    use_chords: bool = False
    use_rests: bool = False
    use_tempos: bool = False
    use_programs: bool = False
    # Special tokens (order matters → maps to ids 0..3).
    special_tokens: tuple[str, ...] = ("PAD", "BOS", "EOS", "MASK")


def settings_for_mode(mode: PerTokMode) -> PerTokSettings:
    """Build the PerTokSettings for one of the three supported modes.

    * ``composition`` — paper Table 1 "PerTok": Pitch, Duration,
      TimeShift, Bar (~196 vocab). For the Cadenza Composer.
    * ``performance`` — paper Table 1 "PerTok-p" (~259) + pedal on/off
      tokens (~261). For the Cadenza Performer.
    * ``legacy_velocity`` — the pre-refactor wrapper config (Velocity but
      NO MicroTime, NO Pedal). Kept only so callers can still load the
      `/dev/shm` cache produced before this refactor.
    """
    if mode == "composition":
        # Composition-only PerTok: no expressive tokens.
        return PerTokSettings(
            use_microtiming=False,
            use_velocities=False,
            use_sustain_pedals=False,
        )
    if mode == "performance":
        # PerTok-p + sustain pedal (user request). The user asked
        # explicitly that the Performer's vocab and PiJAMA caches
        # include sustain-pedal tokens; this is one of the few places
        # Cadenza extends beyond the paper's stated attributes.
        return PerTokSettings(
            use_microtiming=True,
            use_velocities=True,
            use_sustain_pedals=True,
            sustain_pedal_duration=False,            # use Pedal_on / Pedal_off, not durations
        )
    if mode == "legacy_velocity":
        # Pre-refactor settings: PerTok with velocity but no microtime,
        # no pedal. ONLY for loading the existing /dev/shm cache.
        return PerTokSettings(
            use_microtiming=True,                    # legacy had microtiming on
            use_velocities=True,
            use_sustain_pedals=False,
        )
    raise ValueError(f"unknown PerTok mode: {mode!r}")


def _build_pertok(settings: PerTokSettings):
    """Construct the miditok PerTok with our settings. Import is lazy so
    callers that only need the wrapper's metadata avoid the symusic /
    miditok load cost (~250 ms)."""
    from miditok import PerTok, TokenizerConfig                       # type: ignore

    cfg_kwargs = dict(
        pitch_range=settings.pitch_range,
        beat_res={tuple(k): v for k, v in settings.beat_res},
        special_tokens=list(settings.special_tokens),
        use_chords=settings.use_chords,
        use_rests=settings.use_rests,
        use_tempos=settings.use_tempos,
        use_time_signatures=settings.use_time_signatures,
        use_programs=settings.use_programs,
        use_microtiming=settings.use_microtiming,
        ticks_per_quarter=settings.ticks_per_quarter,
        max_microtiming_shift=settings.max_microtiming_shift,
        num_microtiming_bins=settings.num_microtiming_bins,
        num_velocities=settings.num_velocities,
        use_velocities=settings.use_velocities,
        use_sustain_pedals=settings.use_sustain_pedals,
        sustain_pedal_duration=settings.sustain_pedal_duration,
        use_pitchdrum_tokens=settings.use_pitchdrum_tokens,
    )
    return PerTok(TokenizerConfig(**cfg_kwargs))


# ---------------------------------------------------------------------------
# Module-level multiprocessing workers for encode_directory(n_workers>1)
# ---------------------------------------------------------------------------

_PERTOK_WORKER_TOK = None


def _pertok_worker_init(mode: str, drop_bar: bool = True) -> None:
    """Build a per-worker PerTokWrapper instance. Called once per pool worker."""
    global _PERTOK_WORKER_TOK
    # cache_root=None: workers only tokenise; the parent process owns the
    # cache write step.
    _PERTOK_WORKER_TOK = PerTokWrapper.from_default(
        cache_root=None, mode=mode, drop_bar=drop_bar
    )


def _pertok_worker_encode(path_str: str):
    """Encode one MIDI on a pool worker. Returns (path, np.ndarray|Exception|None).

    None means the file was unreadable; the parent will log a [skip].
    """
    try:
        return path_str, _PERTOK_WORKER_TOK.encode_midi(path_str)
    except Exception as e:
        return path_str, e


# ---------------------------------------------------------------------------
# PerTokWrapper
# ---------------------------------------------------------------------------


class PerTokWrapper:
    """Lazy wrapper around miditok.PerTok with caching + decoding helpers.

    The ``mode`` argument controls which paper-aligned PerTok variant is
    built. The mode is also persisted to ``cache_root/meta.json`` so a
    Composer training run cannot accidentally load a Performer cache (or
    vice-versa) without a noisy failure at load time.
    """

    def __init__(
        self,
        settings: Optional[PerTokSettings] = None,
        cache_root: Optional[str | os.PathLike] = None,
        mode: PerTokMode = "composition",
        drop_bar: bool = True,
    ) -> None:
        self.mode: PerTokMode = mode
        self.settings = settings or settings_for_mode(mode)
        self.cache_root = Path(cache_root) if cache_root else None
        self._tokenizer = None  # lazy
        # IDs exposed eagerly so consumers don't have to construct the
        # PerTok object for vocab metadata. Order in ``special_tokens``
        # determines these (PAD=0, BOS=1, EOS=2, MASK=3 by default).
        st = list(self.settings.special_tokens)
        self.pad_id = st.index("PAD") if "PAD" in st else 0
        self.bos_id = st.index("BOS") if "BOS" in st else 1
        self.eos_id = st.index("EOS") if "EOS" in st else 2
        self.mask_id = st.index("MASK") if "MASK" in st else 3
        # Vocab size is determined by PerTok; we materialise it on first
        # access via the lazy ``tokenizer`` property.
        self._vocab_size: Optional[int] = None
        # Bar-token removal at the architecture level (not just at cache
        # build). PiJAMA and Aria-MIDI carry no trustworthy bar grid, so
        # we remove ``Bar_None`` from the effective vocab entirely:
        # caches are stored with COMPACT ids (no slot for Bar), models
        # init with ``vocab_size`` excluding Bar, and decode applies the
        # inverse remap before handing ids to miditok. Result: there is
        # no Bar embedding row, no Bar logit, no Bar id in any training
        # sequence — the architecture cannot represent a bar at all.
        self.drop_bar: bool = drop_bar
        self._bar_ids_orig: Optional[np.ndarray] = None      # original-vocab Bar ids
        self._encode_remap: Optional[np.ndarray] = None       # original_id -> compact_id, -1 for Bar
        self._decode_remap: Optional[np.ndarray] = None       # compact_id -> original_id
        self._orig_vocab_size: Optional[int] = None

    @classmethod
    def from_default(
        cls,
        cache_root: Optional[str | os.PathLike] = None,
        mode: PerTokMode = "composition",
        drop_bar: bool = True,
    ) -> "PerTokWrapper":
        """Factory: return a wrapper for the named PerTok ``mode``.

        Default mode is ``"composition"`` because the Composer is the
        primary training target post-refactor (paper §3.1). Performer
        training must explicitly pass ``mode="performance"``.

        ``drop_bar``: when True (default), the ``Bar_None`` token is
        excised from the vocabulary entirely. The reported ``vocab_size``
        excludes it; encode/decode apply a remap so the model never sees
        a Bar id at any interface. See ``__init__`` docstring.
        """
        return cls(
            settings=settings_for_mode(mode),
            cache_root=cache_root,
            mode=mode,
            drop_bar=drop_bar,
        )

    @property
    def tokenizer(self):
        """Lazily-instantiated miditok PerTok."""
        if self._tokenizer is None:
            self._tokenizer = _build_pertok(self.settings)
            self._orig_vocab_size = int(self._tokenizer.vocab_size)
            self._build_bar_remap()
        return self._tokenizer

    def _build_bar_remap(self) -> None:
        """Compute remap tables that excise ``Bar_None`` (and any other Bar_*
        token miditok might emit) from the model-facing id space.

        Sets:
          self._bar_ids_orig   : original-vocab Bar ids (np.int32)
          self._encode_remap   : np.int32[orig_vocab_size] mapping each
                                 original id to its compact id; entries
                                 for Bar ids are -1 (encoder must filter).
          self._decode_remap   : np.int32[compact_vocab_size] mapping each
                                 compact id back to original (used before
                                 handing ids to miditok decode).
          self._vocab_size     : compact size (= orig - n_bars).
        """
        assert self._tokenizer is not None and self._orig_vocab_size is not None
        vocab = self._tokenizer.vocab  # dict[str, int]
        bar_ids = sorted(
            int(i) for k, i in vocab.items() if str(k).startswith("Bar")
        )
        self._bar_ids_orig = np.array(bar_ids, dtype=np.int32)
        if not self.drop_bar or len(bar_ids) == 0:
            self._encode_remap = np.arange(self._orig_vocab_size, dtype=np.int32)
            self._decode_remap = np.arange(self._orig_vocab_size, dtype=np.int32)
            self._vocab_size = self._orig_vocab_size
            return
        n_orig = self._orig_vocab_size
        bar_set = set(bar_ids)
        encode_remap = np.full(n_orig, -1, dtype=np.int32)
        decode_remap = np.empty(n_orig - len(bar_ids), dtype=np.int32)
        new_id = 0
        for orig_id in range(n_orig):
            if orig_id in bar_set:
                continue
            encode_remap[orig_id] = new_id
            decode_remap[new_id] = orig_id
            new_id += 1
        self._encode_remap = encode_remap
        self._decode_remap = decode_remap
        self._vocab_size = n_orig - len(bar_ids)
        # Specials (PAD/BOS/EOS/MASK) keep their original ids because miditok
        # places them at positions 0..k-1 and bar_id > k. Sanity-check that
        # the remap leaves specials untouched.
        assert encode_remap[self.pad_id] == self.pad_id, "PAD id shifted"
        assert encode_remap[self.mask_id] == self.mask_id, "MASK id shifted"

    @property
    def vocab_size(self) -> int:
        if self._vocab_size is None:
            _ = self.tokenizer    # triggers build + remap
        assert self._vocab_size is not None
        return self._vocab_size

    @property
    def orig_vocab_size(self) -> int:
        """miditok-native vocab size, INCLUDING the Bar slot. Only callers
        that interoperate directly with miditok need this."""
        if self._orig_vocab_size is None:
            _ = self.tokenizer
        assert self._orig_vocab_size is not None
        return self._orig_vocab_size

    # ------------------------------------------------------------------
    # Vocabulary introspection — used by the Performer trainer to find
    # the integer ids of all Velocity / MicroTiming / Pedal tokens so it
    # can build the mask-positions tensor.
    # ------------------------------------------------------------------
    def token_ids_by_prefix(self, prefixes: Sequence[str]) -> list[int]:
        """Return all COMPACT vocab ids whose string starts with any of
        ``prefixes``.

        miditok names PerTok tokens like ``Velocity_60``, ``MicroTime_15``,
        ``Pedal_-1`` / ``PedalOff_-1``. The Performer's training step needs
        the full set of "performance" token ids so it can mask them; this
        helper returns them as a sorted list.

        IDs are returned in the model-facing (compact) id space — i.e.
        Bar tokens are excluded, and ids above the Bar slot are shifted
        down. Callers may safely use the returned ids as embedding-table
        indices.
        """
        vocab = self.tokenizer.vocab     # dict[str, int]
        out: list[int] = []
        for tok_str, tok_id in vocab.items():
            if any(tok_str.startswith(pref) for pref in prefixes):
                # Skip any Bar-prefixed token; specials never start with Bar
                if str(tok_str).startswith("Bar"):
                    continue
                compact_id = int(self._encode_remap[int(tok_id)])
                if compact_id >= 0:
                    out.append(compact_id)
        return sorted(set(out))

    # ------------------------------------------------------------------
    # Encoding / tokenization
    # ------------------------------------------------------------------
    def encode_midi(self, midi_path: str | os.PathLike) -> np.ndarray:
        """Tokenize one MIDI file → np.int32 array of ids in the COMPACT
        (Bar-free) id space.

        miditok still emits Bar events inside the encoder; we filter them
        out here and apply the compact remap so the returned array uses
        the model-facing vocabulary directly. Downstream code (datasets,
        training, sampling) should never see Bar ids."""
        seq = self.tokenizer(str(midi_path))
        # PerTok returns a list of TokSequence (one per track) for multi-track
        # files, or a single TokSequence for one-track. Piano files are
        # single-track in PiJAMA; if multi-track, we take the first track
        # only (the audit spec assumes solo piano).
        if isinstance(seq, list):
            seq = seq[0]
        ids = np.asarray(seq.ids, dtype=np.int32)
        return self._apply_encode_remap(ids)

    def _apply_encode_remap(self, ids: np.ndarray) -> np.ndarray:
        """Map original-vocab ids to compact ids, dropping Bar tokens.

        Pure identity when ``drop_bar=False`` or there are no Bar tokens
        in the vocab, so the hot path adds at most one indexing pass."""
        assert self._encode_remap is not None
        if self._bar_ids_orig is None or len(self._bar_ids_orig) == 0:
            return ids
        # Filter Bar ids in one np.isin pass, then remap the survivors.
        keep = self._encode_remap[ids] >= 0
        return self._encode_remap[ids[keep]].astype(np.int32, copy=False)

    def encode_directory(
        self,
        midi_dir: str | os.PathLike,
        recursive: bool = True,
        suffixes: Sequence[str] = (".mid", ".midi"),
        max_files: Optional[int] = None,
        n_workers: int = 0,
    ) -> tuple[list[np.ndarray], list[str]]:
        """Tokenize all matching files in ``midi_dir``. Returns the list of
        compact-id arrays and a list of (path) strings the same length.

        When ``drop_bar=True`` (default), Bar tokens are excised inside
        ``encode_midi`` itself so every returned array uses the compact
        Bar-free id space. PiJAMA and Aria-MIDI carry no trustworthy bar
        grid (Aria-MIDI is web-sourced with discarded time-signature
        metadata; PiJAMA is solo-piano transcription with no enforced
        bar grid), so the Composer/Performer should not even have an
        embedding row for Bar. The Cadenza paper (Lenz et al. 2024) used
        Lakh-MIDI which carries real bars; we do not.
        """
        root = Path(midi_dir)
        if not root.exists():
            raise FileNotFoundError(f"midi_dir {root} does not exist")
        iterator: Iterable[Path]
        if recursive:
            iterator = (
                p for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in suffixes
            )
        else:
            iterator = (
                p for p in root.iterdir()
                if p.is_file() and p.suffix.lower() in suffixes
            )
        paths = sorted(iterator)
        if max_files is not None:
            paths = paths[:max_files]

        out_ids: list[np.ndarray] = []
        out_paths: list[str] = []
        if self.drop_bar and self._bar_ids_orig is not None and len(self._bar_ids_orig) > 0:
            print(f"  [encode_directory] drop_bar=True: model vocab_size={self.vocab_size} "
                  f"(miditok original {self.orig_vocab_size}; "
                  f"{len(self._bar_ids_orig)} Bar id(s) excised at the tokenizer level)")

        if n_workers > 1 and len(paths) > 100:
            # Multi-process tokenisation — required for the 820k-file
            # Aria-MIDI corpus where single-core is ~22 files/sec ≈ 10h.
            # The worker function and initialiser are module-level (below) so
            # they pickle cleanly under multiprocessing.get_context("spawn").
            import multiprocessing as mp
            mode = getattr(self, "mode", "composition")
            ctx = mp.get_context("spawn")
            paths_str = [str(p) for p in paths]
            n_workers_eff = min(n_workers, max(1, len(paths) // 10))
            print(f"  [encode_directory] multiprocessing with {n_workers_eff} workers "
                  f"over {len(paths):,} files (single-process would take {len(paths)/22/60:.0f} min)")
            with ctx.Pool(
                processes=n_workers_eff,
                initializer=_pertok_worker_init,
                initargs=(mode, self.drop_bar),
            ) as pool:
                done = 0
                last_report_at = 0
                for path_str, result in pool.imap_unordered(
                    _pertok_worker_encode, paths_str, chunksize=32
                ):
                    done += 1
                    if result is None:
                        continue
                    if isinstance(result, Exception):
                        print(f"  [skip] {path_str}: {type(result).__name__}: {result}")
                        continue
                    out_ids.append(result)
                    out_paths.append(path_str)
                    if done - last_report_at >= 10_000:
                        print(f"  [encode_directory] progress: {done:,}/{len(paths):,}")
                        last_report_at = done
        else:
            for p in paths:
                try:
                    ids = self.encode_midi(p)
                except Exception as e:  # broad — PerTok raises various MIDI parse errors
                    print(f"  [skip] {p}: {type(e).__name__}: {e}")
                    continue
                out_ids.append(ids)
                out_paths.append(str(p))

        return out_ids, out_paths

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------
    def cache_paths(self) -> tuple[Path, Path]:
        """Return (npz_path, meta_path) under ``cache_root``."""
        if self.cache_root is None:
            raise RuntimeError("cache_root not set on this wrapper")
        return (
            self.cache_root / "tokens.npz",
            self.cache_root / "meta.json",
        )

    def save_cache(self, all_ids: list[np.ndarray], paths: list[str]) -> tuple[Path, Path]:
        """Save tokenised sequences to a single npz (variable-length via
        np.savez with kw arg per sequence) plus a meta.json carrying paths,
        per-sequence lengths, tokenizer settings, mode, and vocab size.

        Writes are atomic-ish via .tmp+rename so a SIGHUP mid-write can't
        leave a partial cache that the loader treats as authoritative.
        Sequence: write npz.tmp -> write meta.tmp -> rename npz.tmp -> npz
        -> rename meta.tmp -> meta. Loader checks for meta.json; if either
        file is missing or zero-bytes, tokenisation re-runs from scratch.
        """
        if self.cache_root is None:
            raise RuntimeError("cache_root not set on this wrapper")
        self.cache_root.mkdir(parents=True, exist_ok=True)
        npz_path, meta_path = self.cache_paths()
        # numpy.savez auto-appends ".npz" if the filename doesn't end in
        # it, so we must keep the .npz suffix on the tmp name or the
        # later rename target ("tokens.npz.tmp") won't match the
        # actually-written file ("tokens.npz.tmp.npz"). Discovered the
        # hard way at 04:30 UTC on the performance cache: 41 min of
        # tokenisation lost to a typo in the rename target.
        npz_tmp = npz_path.with_name(npz_path.stem + ".tmp" + npz_path.suffix)
        meta_tmp = meta_path.with_name(meta_path.stem + ".tmp" + meta_path.suffix)
        # Variable-length: store as object array.
        arr = np.empty(len(all_ids), dtype=object)
        for i, ids in enumerate(all_ids):
            arr[i] = ids.astype(np.int32, copy=False)
        np.savez_compressed(npz_tmp, sequences=arr, allow_pickle=True)
        meta = {
            "mode": str(self.mode),
            "vocab_size": self.vocab_size,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "mask_id": self.mask_id,
            "settings": _settings_to_jsonable(self.settings),
            "n_sequences": len(all_ids),
            "lengths": [int(x.shape[0]) for x in all_ids],
            "paths": list(paths),
        }
        meta_tmp.write_text(json.dumps(meta, indent=2))
        # Atomic rename (POSIX): npz first, then meta. The Monitor that
        # waits for the cache to be ready should `test -s` BOTH files so
        # it never fires between these two renames.
        npz_tmp.replace(npz_path)
        meta_tmp.replace(meta_path)
        return npz_path, meta_path

    def load_cache(self) -> list[np.ndarray]:
        """Load token sequences from ``cache_root/tokens.npz``. Returns a list
        of int32 arrays.

        If the cache's ``meta.json`` records a different ``mode`` than this
        wrapper, raise — the vocab id space differs across modes, so silently
        loading would produce garbage.
        """
        npz_path, meta_path = self.cache_paths()
        if not npz_path.exists():
            raise FileNotFoundError(f"no cache at {npz_path}")
        # Cross-check mode if meta is present.
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                cache_mode = meta.get("mode")
                if cache_mode is not None and str(cache_mode) != str(self.mode):
                    raise RuntimeError(
                        f"PerTok cache at {self.cache_root} was built with "
                        f"mode={cache_mode!r}, but this wrapper is "
                        f"mode={self.mode!r}. Refusing to load — the vocab "
                        "id spaces differ. Pass the matching mode= to "
                        "PerTokWrapper.from_default()."
                    )
            except json.JSONDecodeError:
                # tolerate corrupt meta — the npz is the source of truth.
                pass
        # ``allow_pickle=True`` because object arrays are pickled.
        loaded = np.load(npz_path, allow_pickle=True)
        seqs = loaded["sequences"]
        # Materialise as a plain list to make downstream slicing simple.
        return [np.asarray(s, dtype=np.int32) for s in seqs]

    def load_meta(self) -> dict:
        _, meta_path = self.cache_paths()
        if not meta_path.exists():
            raise FileNotFoundError(f"no meta at {meta_path}")
        return json.loads(meta_path.read_text())

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def decode_to_symusic(self, token_ids: Sequence[int] | np.ndarray):
        """Decode COMPACT int ids → symusic.Score (native miditok output).

        The ids passed in must use the model-facing (compact, Bar-free)
        vocabulary; we apply the inverse remap before handing them to
        miditok which expects original-vocab ids.
        """
        from miditok.classes import TokSequence                       # type: ignore

        # Touch the tokenizer property first so the remap tables exist.
        _ = self.tokenizer
        compact = np.asarray(token_ids, dtype=np.int32)
        if self.drop_bar and self._decode_remap is not None and len(self._bar_ids_orig or []) > 0:
            # Defensive: any out-of-range id would otherwise crash miditok.
            np.clip(compact, 0, self._vocab_size - 1, out=compact)
            original = self._decode_remap[compact]
        else:
            original = compact
        ids = original.astype(np.int64).tolist()
        ts = TokSequence(ids=ids)
        # Fill the .tokens field (string repr) from .ids.
        self.tokenizer.complete_sequence(ts)
        return self.tokenizer.decode([ts])

    def decode(self, token_ids: Sequence[int] | np.ndarray):
        """Decode int ids → ``pretty_midi.PrettyMIDI`` via
        symusic.Score.dump_midi(bytes) → ``pretty_midi.PrettyMIDI`` round-trip.

        We pick PrettyMIDI as the user-facing output because the rest of
        the project's eval / FMD pipeline (``project3/src/metrics``) is
        PrettyMIDI-based."""
        import pretty_midi                                            # local import
        score = self.decode_to_symusic(token_ids)
        midi_bytes = score.dumps_midi()
        return pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))


def _settings_to_jsonable(s: PerTokSettings) -> dict:
    """asdict() but with the tuple-of-tuple beat_res rendered cleanly."""
    d = asdict(s)
    # asdict turns nested tuples into nested tuples, which json.dump turns
    # into nested lists — fine, but ensure the (lo, hi) keys round-trip.
    return d


__all__ = [
    "PerTokWrapper",
    "PerTokSettings",
    "PerTokMode",
    "settings_for_mode",
]


# ---------------------------------------------------------------------------
# Smoke test — 5 PiJAMA MIDIs round-trip on all three modes.
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    import tempfile

    pijama_root = Path("/var/home/antonio/ispr/project3/data/raw/pijama/midi_hawthorne/midi")
    print(f"== PerTokWrapper smoke test ==")
    if not pijama_root.exists():
        print(f"  PiJAMA root {pijama_root} not present — skipping smoke test")
        return

    # Find 5 MIDI files.
    midis: list[Path] = []
    for p in pijama_root.rglob("*.midi"):
        if p.is_file():
            midis.append(p)
            if len(midis) >= 5:
                break
    if len(midis) < 5:
        print(f"  only found {len(midis)} MIDIs — degraded smoke test")

    for mode in ("composition", "performance", "legacy_velocity"):
        print(f"\n--- mode={mode} ---")
        with tempfile.TemporaryDirectory() as tmp:
            tok = PerTokWrapper.from_default(cache_root=tmp, mode=mode)
            print(f"  vocab_size       : {tok.vocab_size}")
            print(f"  pad/bos/eos/mask : "
                  f"{tok.pad_id} / {tok.bos_id} / {tok.eos_id} / {tok.mask_id}")
            if mode == "performance":
                # Spot-check that pedal tokens are in the vocab.
                pedal_ids = tok.token_ids_by_prefix(["Pedal"])
                print(f"  pedal token ids   : {pedal_ids}")
                vel_ids = tok.token_ids_by_prefix(["Velocity"])
                # miditok emits the token as "MicroTiming_<bin>", not
                # "MicroTime_<bin>" — match the exact string the
                # PerTok class uses internally.
                micro_ids = tok.token_ids_by_prefix(["MicroTiming"])
                print(f"  n velocity tokens : {len(vel_ids)}")
                print(f"  n microtime tokens: {len(micro_ids)}")

            ids_list = []
            paths_list = []
            for m in midis:
                ids = tok.encode_midi(m)
                ids_list.append(ids)
                paths_list.append(str(m))
                # Quick id-range sanity.
                assert ids.min() >= 0 and ids.max() < tok.vocab_size

            # Save / load round-trip.
            npz_path, meta_path = tok.save_cache(ids_list, paths_list)
            loaded = tok.load_cache()
            assert len(loaded) == len(ids_list)
            for a, b in zip(ids_list, loaded):
                assert (a == b).all(), "cache round-trip mismatch"
            print(f"  cache round-trip : ok  ({len(loaded)} sequences)")

            # Cross-mode load must FAIL (id spaces differ).
            try:
                wrong = PerTokWrapper.from_default(
                    cache_root=tmp,
                    mode=("performance" if mode != "performance" else "composition"),
                )
                wrong.load_cache()
            except RuntimeError as e:
                print(f"  cross-mode refused: ok ({type(e).__name__})")
            else:
                raise AssertionError(
                    "cross-mode load should have raised RuntimeError"
                )

            # Decode one back to PrettyMIDI for the eval pipeline.
            midi = tok.decode(ids_list[0])
            print(
                f"  decode → PrettyMIDI: "
                f"{len(midi.instruments)} instr, "
                f"{sum(len(i.notes) for i in midi.instruments)} notes"
            )
    print("\n== PerTokWrapper smoke test passed ==")


if __name__ == "__main__":
    _smoke_test()
