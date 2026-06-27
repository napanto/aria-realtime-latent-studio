#!/usr/bin/env python3
#
# This file is vendored from EleutherAI/aria
# (https://github.com/EleutherAI/aria) and is licensed under the Apache
# License 2.0; see the LICENSE file at the repository root for the full text.
# It is used here as the real-time MLX decoding engine.
#

import argparse
import os
import time
import uuid
import random
import logging
import threading
import queue
import math
import sys
import pathlib
import select
import json
import mido

import mlx.core as mx
import mlx.nn as nn

from ariautils.midi import MidiDict, midi_to_dict
from ariautils.tokenizer import AbsTokenizer
from aria.inference.model_mlx import TransformerLM
from aria.model import ModelConfig
from aria.config import load_model_config
from aria.run import _get_embedding

EMBEDDING_OFFSET: int = 0

DTYPE = mx.bfloat16
MAX_SEQ_LEN: int = 4096
KV_CHUNK_SIZE: int = 256
PREFILL_CHUNK_SIZE_L: int = 128
PREFILL_CHUNK_SIZE: int = 16
RECALC_DUR_PREFILL_CHUNK_SIZE: int = 8
RECALC_DUR_BUFFER_MS: int = 100

BEAM_WIDTH: int = 3
TIME_TOK_WEIGHTING: int = -5
FIRST_ONSET_BUFFER_MS: int = -150
# Lag (ms) tolerated before the stream "stretches" (re-anchors) instead of
# playing a note immediately. NOT a drop threshold anymore — notes are never
# dropped; this just bounds how late a note may be before the timeline shifts.
MAX_STREAM_DELAY_MS: int = int(os.environ.get("ARIA_STREAM_DELAY_MS", "200"))
# How far behind real-time the live stream currently is (ms). Updated by
# stream_midi as it stretches the timeline; read by the engine -> /api/realtime/
# status -> GUI so the player can see when the (slow) decode is falling behind.
CURRENT_STREAM_LAG_MS: float = 0.0
# Phrase-response mode: a fixed playback LEAD (ms). Set >0 to trail generation
# by this much so a slow decode buffers the response ahead and plays it back
# smoothly (no pauses/drops), at the cost of an initial delay. 0 = live
# streaming (play as generated).
PHRASE_LEAD_MS: int = 0

# Turn-switch timing (when you hand over with the takeover CC). Three modes:
#   "snap"    - original: the model jumps its clock to wall-clock, so the elapsed
#               real time (your pause + prefill) becomes a rest before its first
#               note, which lands off your rhythmic grid.
#   "freeze"  - the clock is frozen at takeover: the model continues on-grid from
#               just after your last note, its whole continuation shifted later
#               by ~TURN_FREEZE_LATENCY_MS. No rest, but its phase restarts at
#               the handover.
#   "catchup" - the model continues with no rest AND generates through the
#               switching pause, DISCARDING the notes that fall in the past, so
#               its first audible note lands on your ORIGINAL tempo grid at
#               wall-clock now (phase-locked to your playing). Costs a little
#               extra generation to fast-forward through the discarded notes.
# Applies to turn-taking only (duet manages its own timing).
TURN_SWITCH_MODE: str = "catchup"
TURN_FREEZE_LATENCY_MS: int = 400  # used by "freeze" only

# Duet mode: after a burst, hold the capture window open until the model's
# look-ahead notes have played (+buffer), capped so a far-ahead burst can't
# stall the loop.
DUET_CATCHUP_BUFFER_MS: int = 50
DUET_MAX_CATCHUP_MS: int = 4000

MIN_NOTE_DELTA_MS: int = 0
MIN_PEDAL_DELTA_MS: int = 0
MIN_NOTE_LENGTH_MS: int = 10
# Upper bound on how long a note-off may be deferred to preserve its intended
# audible duration when the decode falls behind real-time (see stream_midi).
# The AbsTokenizer caps a single note duration at 5000ms, so this never clips a
# legitimate note; it only guards against a malformed token group.
MAX_NOTE_HOLD_MS: int = 5000
HARDWARE_INPUT_LATENCY_MS: int = 0
BASE_OUTPUT_LATENCY_MS: int = 0
VELOCITY_OUTPUT_LATENCY_MS: dict[int, int] = {v: 0 for v in range(0, 127, 10)}


TOKENIZER_CONFIG_PATH = (
    pathlib.Path(__file__)
    .parent.resolve()
    .joinpath("demo-tokenizer-config.json")
)


_persistent_virtual_ports: dict = {}


class _KeepAlive:
    """Context-manager wrapper that exposes a persistent port to a `with`
    block without closing it on exit, so a virtual port stays visible to
    other apps (e.g. Pianoteq) for the whole session rather than only while
    a `with mido.open_output(...)` block is active."""

    def __init__(self, port):
        self._port = port

    def __enter__(self):
        return self._port

    def __exit__(self, *exc):
        return False  # keep the port open


def open_output(name: str):
    """Open an existing MIDI output port by name, or create it as a virtual
    port if no hardware/IAC port of that name exists. Lets you route the
    model's output straight into a softsynth (e.g. Pianoteq) without setting
    up an IAC bus in Audio MIDI Setup. Virtual ports are created once and kept
    open for the lifetime of the process so they remain visible between turns."""
    if name in mido.get_output_names():
        return mido.open_output(name)
    if name not in _persistent_virtual_ports:
        _persistent_virtual_ports[name] = mido.open_output(name, virtual=True)
    return _KeepAlive(_persistent_virtual_ports[name])
file_handler = logging.FileHandler("./demo.log", mode="w")
file_handler.setLevel(logging.DEBUG)


def get_logger(name: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.propagate = False
        logger.setLevel(logging.DEBUG)

        class MillisecondFormatter(logging.Formatter):
            def formatTime(self, record, datefmt=None):
                created_ms = int(record.created * 1000)
                return str(created_ms)

        if name is not None:
            formatter = MillisecondFormatter(
                "%(asctime)s: [%(levelname)s] [%(name)s] %(message)s"
            )
        else:
            formatter = MillisecondFormatter(
                "%(asctime)s: [%(levelname)s] %(message)s"
            )

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def parse_args():
    argp = argparse.ArgumentParser()
    argp.add_argument("--checkpoint", help="path to model checkpoint")
    argp.add_argument("--midi_in", required=False, help="MIDI input port")
    argp.add_argument(
        "--control_midi_in",
        required=False,
        help="Separate MIDI input port for control signals only (e.g. a "
        "Morningstar MC8 foot controller). Messages from this port are routed "
        "to the control queue but never tokenized as performance input.",
    )
    argp.add_argument("--midi_out", required=True, help="MIDI output port")
    argp.add_argument(
        "--midi_through",
        required=False,
        help="MIDI through port for received input",
    )
    argp.add_argument(
        "--midi_path",
        required=False,
        help="Use MIDI file instead of MIDI input port",
    )
    argp.add_argument(
        "--midi_control_signal",
        type=int,
        help="MIDI control change message for AI takeover",
    )
    argp.add_argument(
        "--midi_reset_control_signal",
        type=int,
        help="MIDI control change message context reset",
    )
    argp.add_argument(
        "--back_and_forth",
        action="store_true",
        help="Enable toggling between human and AI. If not set, the control signal will reset the session.",
        required=False,
    )
    argp.add_argument(
        "--temp",
        help="sampling temperature value",
        type=float,
        required=False,
        default=0.95,
    )
    argp.add_argument(
        "--min_p",
        help="sampling min_p value",
        type=float,
        required=False,
        default=0.03,
    )
    argp.add_argument(
        "--wait_for_close",
        help="wait for note-offs before generating",
        action="store_true",
    )
    argp.add_argument(
        "--max_seq_len",
        type=int,
        default=8192,
        help="Model context window in tokens (default 8192, the model's trained "
        "max). A single continuous generation ends when it fills this. Larger = "
        "longer before it stops, but more KV-cache memory; lower it if the 8GB "
        "machine runs out of memory.",
    )
    argp.add_argument(
        "--turn_switch_mode",
        choices=["snap", "freeze", "catchup"],
        default="catchup",
        help="Turn-switch timing (turn-taking only): 'snap' inserts the elapsed "
        "pause as a rest (first note off-grid); 'freeze' continues on-grid from "
        "your last note, shifted later by --turn_freeze_latency_ms; 'catchup' "
        "(default) continues with no rest and discards the notes that fall in "
        "the switching pause so it joins on your original tempo grid.",
    )
    argp.add_argument(
        "--turn_freeze_latency_ms",
        type=int,
        default=400,
        help="Latency (ms) the 'freeze' turn-switch mode shifts the model's "
        "continuation by (default 400). Ignored by other modes.",
    )
    argp.add_argument(
        "--duet",
        action="store_true",
        help="Experimental: play together with the model in a tight interleave "
        "loop (no turn hand-off). Both your notes and the model's accumulate in "
        "the shared context.",
    )
    argp.add_argument(
        "--duet_listen_ms",
        type=int,
        default=250,
        help="Duet mode: length of each capture window in ms (default 250).",
    )
    argp.add_argument(
        "--duet_play_ms",
        type=int,
        default=500,
        help="Duet mode: ms of token generation per burst, on top of the "
        "~0.85s prefill floor on an M1 (default 500).",
    )
    argp.add_argument(
        "--quantize",
        help="apply model quantize",
        action="store_true",
    )
    argp.add_argument(
        "--save_path",
        type=str,
        required=False,
        help="path to save complete MIDI file",
    )
    argp.add_argument(
        "--hardware",
        type=str,
        required=False,
        help="path to json file containing hardware calibration settings",
    )
    argp.add_argument(
        "--embedding_checkpoint",
        type=str,
        help="path to embedding model checkpoint for conditioned generation",
        required=False,
    )
    argp.add_argument(
        "--embedding_midi_path",
        type=str,
        help="path to embedding MIDI file for conditioned generation",
        required=False,
    )
    argp.add_argument(
        "--playback",
        action="store_true",
        help="playback file at midi_path through output_port",
        required=False,
    )

    return argp.parse_args()


def set_calibration_settings(load_path: str):
    with open(load_path, "r") as f:
        _settings = json.load(f)

    global MIN_NOTE_DELTA_MS
    global MIN_PEDAL_DELTA_MS
    global MIN_NOTE_LENGTH_MS
    global HARDWARE_INPUT_LATENCY_MS
    global BASE_OUTPUT_LATENCY_MS
    global VELOCITY_OUTPUT_LATENCY_MS

    MIN_NOTE_DELTA_MS = _settings["MIN_NOTE_DELTA_MS"]
    MIN_PEDAL_DELTA_MS = _settings["MIN_PEDAL_DELTA_MS"]
    MIN_NOTE_LENGTH_MS = _settings["MIN_NOTE_LENGTH_MS"]
    HARDWARE_INPUT_LATENCY_MS = _settings["HARDWARE_INPUT_LATENCY_MS"]
    BASE_OUTPUT_LATENCY_MS = _settings["BASE_OUTPUT_LATENCY_MS"]
    VELOCITY_OUTPUT_LATENCY_MS = {
        int(k): v for k, v in _settings["VELOCITY_OUTPUT_LATENCY_MS"].items()
    }


def _get_input_latency_ms(velocity: int):
    return BASE_OUTPUT_LATENCY_MS + VELOCITY_OUTPUT_LATENCY_MS[velocity]


def get_epoch_time_ms() -> int:
    return round(time.time() * 1000)


def prefill(
    model: TransformerLM,
    idxs: mx.array,
    input_pos: mx.array,
) -> mx.array:
    # pad_idxs is only needed for prepended pad tokens
    ip = input_pos + EMBEDDING_OFFSET
    mkp = math.ceil((input_pos[-1].item() + EMBEDDING_OFFSET) / KV_CHUNK_SIZE) * KV_CHUNK_SIZE
    off = input_pos[0] + EMBEDDING_OFFSET
    return model(idxs=idxs, input_pos=ip, max_kv_pos=mkp, offset=off)


def decode_one(
    model: TransformerLM,
    idxs: mx.array,
    input_pos: mx.array,
) -> mx.array:
    assert input_pos.shape[-1] == 1

    ip = input_pos + EMBEDDING_OFFSET
    mkp = math.ceil((input_pos[-1].item() + EMBEDDING_OFFSET) / KV_CHUNK_SIZE) * KV_CHUNK_SIZE
    off = input_pos[0] + EMBEDDING_OFFSET
    return model(idxs=idxs, input_pos=ip, max_kv_pos=mkp, offset=off)[:, -1]


def sample_min_p(logits: mx.array, p_base: float):
    """Min_p sampler in logit space, see - https://arxiv.org/pdf/2407.01082"""
    if p_base <= 0.0:
        return mx.argmax(logits, axis=-1, keepdims=True)
    if p_base >= 1.0:
        return mx.random.categorical(logits, num_samples=1)

    log_p_max = mx.max(logits, axis=-1, keepdims=True)
    log_p_scaled = mx.log(p_base) + log_p_max
    mask = logits >= log_p_scaled
    masked_logits = mx.where(~mask, -mx.inf, logits)
    next_token = mx.random.categorical(masked_logits, num_samples=1)

    return next_token


def _warmup_prefill(
    model: TransformerLM,
    logger: logging.Logger,
    chunk_size: int,
):
    assert chunk_size > 1

    compile_start_time_s = time.time()
    logger.info(f"Compiling prefill (chunk_size={chunk_size})")
    for idx in range(8):
        start = idx * (MAX_SEQ_LEN - chunk_size) // 7
        mx.eval(
            prefill(
                model,
                idxs=mx.ones([1, chunk_size], dtype=mx.int32),
                input_pos=mx.arange(
                    start,
                    start + chunk_size,
                    dtype=mx.int32,
                ),
            )
        )

    logger.info(
        f"Finished compiling - took {time.time() - compile_start_time_s:.4f} seconds"
    )

    bench_start_time_s = time.time()
    mx.eval(
        prefill(
            model,
            idxs=mx.ones([1, chunk_size], dtype=mx.int32),
            input_pos=mx.arange(0, chunk_size, dtype=mx.int32),
        )
    )
    bench_end_time_s = time.time()
    bench_ms = 1e3 * (bench_end_time_s - bench_start_time_s)
    bench_its = 1000 / bench_ms
    logger.info(
        f"Compiled prefill benchmark: {bench_ms:.2f} ms/it ({bench_its:.2f} it/s)"
    )

    return model


def _warmup_decode_one(
    model: TransformerLM,
    logger: logging.Logger,
):
    # Don't need to explicitly compile with mlx, instead we are just precalculating
    # the computation graphs for different shapes
    compile_start_time_s = time.time()
    for _ in range(5):
        mx.eval(
            decode_one(
                model,
                idxs=mx.array([[random.randint(0, 20)]], dtype=mx.int32),
                input_pos=mx.array([MAX_SEQ_LEN - 1], dtype=mx.int32),
            ),
        )
    logger.info(
        f"Finished compiling - took {time.time() - compile_start_time_s:.4f} seconds"
    )

    bench_start_time_s = time.time()
    mx.eval(
        decode_one(
            model,
            idxs=mx.array([[0]], dtype=mx.int32),
            input_pos=mx.array([0], dtype=mx.int32),
        )
    )
    bench_end_time_s = time.time()
    bench_ms = 1e3 * (bench_end_time_s - bench_start_time_s)
    bench_its = 1000 / bench_ms
    logger.info(
        f"Compiled decode_one benchmark: {bench_ms:.2f} ms/it ({bench_its:.2f} it/s)"
    )

    return model


def warmup_model(model: TransformerLM):
    logger = get_logger()

    model.eval()
    model.setup_cache(
        batch_size=1,
        max_seq_len=MAX_SEQ_LEN,
        dtype=DTYPE,
    )

    model = _warmup_decode_one(model=model, logger=logger)
    for chunk_size in list(
        {
            PREFILL_CHUNK_SIZE,
            RECALC_DUR_PREFILL_CHUNK_SIZE,
        }
    ):
        model = _warmup_prefill(
            model=model, logger=logger, chunk_size=chunk_size
        )

    return model


def load_model(checkpoint_path: str):
    logger = get_logger()

    tokenizer = AbsTokenizer(config_path=TOKENIZER_CONFIG_PATH)
    model_config = ModelConfig(**load_model_config("medium-emb"))
    model_config.set_vocab_size(tokenizer.vocab_size)

    weights = mx.load(checkpoint_path)
    for key, weight in weights.items():
        if weight.dtype != DTYPE:
            weights[key] = weight.astype(DTYPE)

    logging.info(f"Loading model weights from {checkpoint_path}")

    init_start_time_s = time.time()
    model = TransformerLM(model_config)

    assert (
        tokenizer.vocab_size == weights["model.tok_embeddings.weight"].shape[0]
    ), (
        "Embedding shape mismatch. Ensure that you are loading the "
        f"demo-specific checkpoint. tokenizer={tokenizer.vocab_size}, "
        f"weights={weights['model.tok_embeddings.weight'].shape[0]}"
    )

    model.load_weights(list(weights.items()), strict=False)
    model.eval()

    if args.quantize:
        nn.quantize(model.model, group_size=32, bits=8)

    logger.info(
        f"Finished initializing model - took {time.time() - init_start_time_s:.4f} seconds"
    )

    return model


def _first_bad_dur_index(
    tokenizer: AbsTokenizer,
    priming_seq: list,
    pred_ids: list,
    chunk_start: int,
    last_offset_ms: int,
    logger: logging.Logger,
):
    num_time_toks = priming_seq[:chunk_start].count(tokenizer.time_tok)
    local_onset_ms = tokenizer.calc_length_ms(
        priming_seq[: chunk_start + 1], onset=True
    )  # chunk_start + 1 to account for possibly truncated dur token
    logger.debug(f"Starting from local onset {local_onset_ms}")

    for pos, tok_id in enumerate(
        pred_ids[: len(priming_seq) - chunk_start], start=chunk_start
    ):
        prim_tok = priming_seq[pos]  # Should never error?
        pred_tok = tokenizer.id_to_tok[tok_id]
        logger.debug(f"prim={prim_tok}, pred={pred_tok}")

        if isinstance(prim_tok, tuple) and prim_tok[0] == "onset":
            local_onset_ms = num_time_toks * 5000 + prim_tok[1]
        elif prim_tok == tokenizer.time_tok:
            num_time_toks += 1
        elif isinstance(prim_tok, tuple) and prim_tok[0] == "dur":
            # The model's prediction at a dur position is not guaranteed to be a
            # dur token (it can predict anything). Only resample when it actually
            # predicted a longer duration; otherwise skip this position.
            if isinstance(pred_tok, tuple) and pred_tok[0] == "dur":
                dur_true = prim_tok[1]
                dur_pred = pred_tok[1]
                if dur_pred > dur_true and (
                    local_onset_ms + dur_true
                    >= last_offset_ms - RECALC_DUR_BUFFER_MS
                ):
                    logger.info(
                        f"Found token to resample at {pos}: {prim_tok} -> {pred_tok}"
                    )
                    return pos

    return None


def recalc_dur_tokens_chunked(
    model: TransformerLM,
    priming_seq: list,
    enc_seq: mx.array,
    tokenizer: AbsTokenizer,
    start_idx: int,
):
    # Speculative-decoding inspired duration re-calculation
    assert start_idx > 0
    logger = get_logger("GENERATE")

    priming_len = len(priming_seq)
    last_offset = tokenizer.calc_length_ms(priming_seq, onset=False)
    logger.debug(
        f"Using threshold for duration recalculation: {last_offset - RECALC_DUR_BUFFER_MS}"
    )

    idx = start_idx
    while idx <= priming_len:
        # Clamp to the priming length. Overshooting reads UNINITIALISED enc_seq
        # positions (>= priming_len) and prefills them into the KV cache; the
        # first generated token then attends to that garbage -> NaN logits ->
        # the turn-switch crash / pedal-spam / silence. Never go past priming.
        end_idx = min(idx + RECALC_DUR_PREFILL_CHUNK_SIZE, priming_len + 1)

        window_ids = mx.array(
            enc_seq[:, idx - 1 : end_idx - 1].tolist(),
            dtype=mx.int32,
        )
        window_pos = mx.arange(idx - 1, end_idx - 1, dtype=mx.int32)

        logger.info(
            f"Recalculating chunked durations for positions: {idx - 1} - {end_idx - 2}"
        )

        logits = prefill(model, idxs=window_ids, input_pos=window_pos)
        if bool(mx.isnan(logits).any().item()):
            logger.warning(
                f"[DECODE] NaN in recalc prefill logits at positions "
                f"{idx - 1}-{end_idx - 2} (ids={window_ids[0].tolist()})"
            )
        pred_ids = mx.argmax(logits, axis=-1).flatten().tolist()

        logger.debug(f"Inserted: {tokenizer.decode(window_ids[0].tolist())}")
        logger.debug(f"Positions: {window_pos.tolist()}")
        logger.debug(f"Predictions: {tokenizer.decode(pred_ids)}")

        bad_pos = _first_bad_dur_index(
            tokenizer=tokenizer,
            priming_seq=priming_seq,
            pred_ids=pred_ids,
            chunk_start=idx,
            last_offset_ms=last_offset,
            logger=logger,
        )

        if bad_pos is None:
            idx = end_idx
        else:
            new_id = pred_ids[bad_pos - idx]
            enc_seq[0, bad_pos] = new_id
            priming_seq[bad_pos] = tokenizer.id_to_tok[new_id]
            idx = bad_pos + 1

    next_logits = logits[:, priming_len - idx]

    _kv_ctx = model.get_kv_ctx()
    if _kv_ctx is not None:  # None in some catchup paths; don't crash a debug line
        logger.debug(f"Internal KV-state: {tokenizer.decode(_kv_ctx)}")

    return enc_seq, priming_seq, next_logits


def decode_first_tokens(
    model: TransformerLM,
    first_token_logits: mx.array,
    enc_seq: mx.array,
    priming_seq: list,
    tokenizer: AbsTokenizer,
    generated_tokens_queue: queue.Queue,
    first_on_msg_epoch_ms: int,
    catchup_discard: bool = False,
):
    logger = get_logger("GENERATE")

    # buffer_ms determines how far in the past to start generating notes.
    buffer_ms = FIRST_ONSET_BUFFER_MS
    time_tok_id = tokenizer.tok_to_id[tokenizer.time_tok]
    eos_tok_id = tokenizer.tok_to_id[tokenizer.eos_tok]
    dim_tok_id = tokenizer.tok_to_id[tokenizer.dim_tok]
    ped_off_id = tokenizer.tok_to_id[tokenizer.ped_off_tok]
    # Structural special tokens that must never be generated (incl. the
    # delimiter <X>); <T> (time) and pedal tokens stay valid.
    structural_mask_ids = [
        tokenizer.tok_to_id[t]
        for t in (
            tokenizer.bos_tok,
            tokenizer.pad_tok,
            tokenizer.unk_tok,
            tokenizer.dim_tok,
            tokenizer.delimiter_tok,
        )
    ]

    logits = first_token_logits
    time_since_first_onset_ms = get_epoch_time_ms() - first_on_msg_epoch_ms
    idx = len(priming_seq) + 1

    num_time_toks_required = (time_since_first_onset_ms + buffer_ms) // 5000
    num_time_toks_in_priming_seq = priming_seq.count(tokenizer.time_tok)
    num_time_toks_to_add = num_time_toks_required - num_time_toks_in_priming_seq

    if catchup_discard:
        # Catchup mode: do NOT force the model's clock up to wall-clock. It
        # continues from the last note; the elapsed pause is generated as real
        # notes (later discarded as stale by stream_midi) rather than inserted
        # as a rest. So suppress catch-up time tokens here.
        num_time_toks_to_add = 0

    logger.info(f"Time since first onset: {time_since_first_onset_ms}ms")
    logger.info(f"Using first note-onset buffer: {buffer_ms}ms")

    while num_time_toks_to_add > 0:
        generated_tokens_queue.put(tokenizer.time_tok)
        logits = decode_one(
            model,
            idxs=mx.array([[time_tok_id]], dtype=mx.int32),
            input_pos=mx.array([idx - 1], dtype=mx.int32),
        )

        logger.info(f"Inserted time_tok at position {idx - 1}")
        num_time_toks_to_add -= 1
        enc_seq[:, idx - 1] = time_tok_id
        idx += 1

    # Defend against NaN logits from a corrupted decode step (intermittent on
    # some pedal-heavy contexts): NaN -> -inf so log_softmax stays finite and the
    # beam can still pick a valid token instead of producing NaN scores (which
    # crashed the thread). Logged so the corruptor stays visible.
    if bool(mx.isnan(logits).any().item()):
        logger.warning("[DECODE] NaN in first-token logits; sanitising to recover")
        logits = mx.where(
            mx.isnan(logits), mx.array(-float("inf"), dtype=logits.dtype), logits
        )
        if not bool(mx.isfinite(logits).any().item()):
            logits = mx.zeros_like(logits)  # all non-finite -> uniform fallback

    logits[:, structural_mask_ids] = float("-inf")
    logits[:, tokenizer.tok_to_id[tokenizer.eos_tok]] = float("-inf")
    logits[:, tokenizer.tok_to_id[tokenizer.ped_off_tok]] = float("-inf")

    # MLX doesn't have a equivalent of torch topk
    log_probs = nn.log_softmax(logits, axis=-1)
    top_ids = mx.argsort(log_probs, axis=-1)[0, -BEAM_WIDTH:]
    top_log_probs = log_probs[0, top_ids]

    # top_log_probs are sorted in ascending order
    if time_tok_id not in top_ids.tolist():
        top_ids[0] = time_tok_id
        top_log_probs[0] = log_probs[0, time_tok_id]

    _time_tok_idx = top_ids.tolist().index(time_tok_id)
    top_log_probs[_time_tok_idx] += TIME_TOK_WEIGHTING

    top_toks = [tokenizer.id_to_tok[id] for id in top_ids.tolist()]

    logger.debug(f"Calculated top {BEAM_WIDTH} tokens={top_toks}")
    logger.debug(f"Calculated top {BEAM_WIDTH} scores={top_log_probs.tolist()}")

    priming_seq_last_onset_ms = tokenizer.calc_length_ms(
        priming_seq, onset=True
    )

    if catchup_discard:
        # Allow onsets "in the past": the model continues from the last note and
        # those early notes are dropped downstream, so they must not be masked.
        masked_onset_ids = []
    elif priming_seq_last_onset_ms < time_since_first_onset_ms + buffer_ms:
        masked_onset_ids = [
            tokenizer.tok_to_id[tok]
            for tok in tokenizer.onset_tokens
            if tok[1] < ((time_since_first_onset_ms + buffer_ms) % 5000)
        ]

    else:
        masked_onset_ids = []

    logger.debug(
        f"Masking onsets for {len(masked_onset_ids)} tokens ({time_since_first_onset_ms + buffer_ms})"
    )

    # Fallbacks so a degenerate/NaN logit step can't leave these unbound: the
    # beam's `score > best_score` never fires when every candidate scores NaN
    # (NaN comparisons are always False), which otherwise crashes the thread on
    # the `best_tok_1` reference below. Default to advancing time (<T>) — always
    # grammatical and keeps the stream alive instead of dying silently.
    best_tok_id_1 = best_tok_id_2 = time_tok_id
    best_tok_1 = best_tok_2 = tokenizer.id_to_tok[time_tok_id]
    best_score = float("-inf")
    for i in range(BEAM_WIDTH):
        tok = top_toks[i]
        tok_id = top_ids[i].item()
        tok_log_prob = top_log_probs[i]

        next_logits = decode_one(
            model,
            idxs=mx.array([[tok_id]], dtype=mx.int32),
            input_pos=mx.array([idx - 1], dtype=mx.int32),
        )
        logger.debug(
            f"Sampled logits for positions {idx} by inserting {tok} at position {idx - 1}"
        )

        next_log_probs = nn.log_softmax(next_logits, axis=-1)

        next_log_probs[:, eos_tok_id] = float("-inf")
        next_log_probs[:, structural_mask_ids] = float("-inf")
        next_log_probs[:, ped_off_id] = float("-inf")

        if masked_onset_ids:
            next_log_probs[:, masked_onset_ids] = float("-inf")
        if tok_id == time_tok_id:
            next_log_probs[:, time_tok_id] = float("-inf")

        next_tok_log_prob = mx.max(next_log_probs, axis=-1)
        next_tok_id = mx.argmax(next_log_probs, axis=-1)
        next_tok = tokenizer.id_to_tok[next_tok_id.item()]
        score = float((tok_log_prob + next_tok_log_prob).item())

        logger.info(
            f"Calculated tuple {(tok, next_tok)} with scores {(tok_log_prob.item(), next_tok_log_prob.item())} (combined={score})"
        )

        # `score == score` rejects NaN (NaN != NaN) so a bad step can't win.
        # best_score stays a plain float (fallback -inf) so the log below never
        # calls .item() on a float.
        if score == score and score > best_score:
            best_tok_id_1, best_tok_id_2 = tok_id, next_tok_id.item()
            best_tok_1, best_tok_2 = (
                tokenizer.id_to_tok[best_tok_id_1],
                tokenizer.id_to_tok[best_tok_id_2],
            )
            best_score = score

    logger.info(
        f"Chose tuple {(best_tok_1, best_tok_2)} with score {best_score}"
    )

    enc_seq[:, idx - 1] = best_tok_id_1
    enc_seq[:, idx] = best_tok_id_2
    generated_tokens_queue.put(tokenizer.id_to_tok[best_tok_id_1])
    generated_tokens_queue.put(tokenizer.id_to_tok[best_tok_id_2])

    mx.eval(
        decode_one(
            model,
            idxs=mx.array([[best_tok_id_1]], dtype=mx.int32),
            input_pos=mx.array([idx - 1], dtype=mx.int32),
        )
    )

    logger.info(
        f"Updated KV-Cache by re-inserting {best_tok_1} at position {idx - 1}"
    )
    _kv_ctx = model.get_kv_ctx()
    if _kv_ctx is not None:  # None in some catchup paths; don't crash a debug line
        logger.debug(f"Internal KV-state: {tokenizer.decode(_kv_ctx)}")

    return enc_seq, idx + 1


# Grammar-constrained realtime decoding ---------------------------------------
# The realtime decoder masks structural/short-dur tokens by POSITION, which still
# lets the model emit out-of-grammar groups (e.g. two notes with no onset between
# them) that the note decoder must then drop ("Skipping malformed token group").
# Layering the note/pedal FSM in studio/mlx_vae/grammar.py makes every sampled
# token grammatically valid by construction, so malformed groups never occur.
# Degrades to structural-only masking if the grammar module can't be imported —
# never breaks generation.
_GRAMMAR_CACHE: dict = {}


def _build_grammar_cached(tokenizer):
    # Escape hatch / A-B switch: ARIA_RT_NO_GRAMMAR=1 disables the FSM entirely
    # (falls back to structural-only masking) without a redeploy.
    if os.environ.get("ARIA_RT_NO_GRAMMAR"):
        return (None, None)
    # Key on (id, vocab_size) so a recycled Python id for a different-vocab
    # tokenizer (after an engine restart) can't return a stale grammar.
    key = (id(tokenizer), int(getattr(tokenizer, "vocab_size", 0)))
    if key not in _GRAMMAR_CACHE:
        grammar, fsm_cls = None, None
        try:
            gdir = str(
                pathlib.Path(__file__).resolve().parent.parent / "studio" / "mlx_vae"
            )
            if gdir not in sys.path:
                sys.path.insert(0, gdir)
            from grammar import build_grammar, GrammarFSM

            grammar = build_grammar(tokenizer)
            fsm_cls = GrammarFSM
        except Exception as e:  # noqa: BLE001 — degrade, never break decoding
            get_logger("GENERATE").warning(
                f"grammar FSM unavailable ({e!r}); structural masking only"
            )
        _GRAMMAR_CACHE[key] = (grammar, fsm_cls)
    return _GRAMMAR_CACHE[key]


def decode_tokens(
    model: TransformerLM,
    enc_seq: mx.array,
    tokenizer: AbsTokenizer,
    control_sentinel: threading.Event,
    generated_tokens_queue: queue.Queue,
    idx: int,
    temperature: float,
    min_p: float,
    is_ending: bool,
    max_gen_ms: int | None = None,
):
    logger = get_logger("GENERATE")
    logger.info(
        f"Using sampling parameters: temperature={temperature}, min_p={min_p}"
    )

    if control_sentinel.is_set():
        control_sentinel.clear()

    # Duet mode bounds each burst by wall-clock time rather than by an external
    # control signal. The deadline is measured from the first generated token
    # (decode_tokens runs *after* prefill/first-token), so it caps actual token
    # production, not the unavoidable prefill latency.
    deadline_ms = (
        get_epoch_time_ms() + max_gen_ms if max_gen_ms is not None else None
    )

    last_tok_is_pedal = False
    dur_ids = [tokenizer.tok_to_id[idx] for idx in tokenizer.dur_tokens]
    dur_mask_ids = [
        tokenizer.tok_to_id[("dur", dur_ms)]
        for dur_ms in range(0, MIN_NOTE_LENGTH_MS, 10)
    ]
    # Structural special tokens that must never appear inside the generated
    # stream -- they break the note decoder. High temperature / low min_p can
    # otherwise sample them (e.g. the delimiter <X>). <T> (time) and the pedal
    # tokens stay valid; eos is handled separately below.
    structural_mask_ids = [
        tokenizer.tok_to_id[t]
        for t in (
            tokenizer.bos_tok,
            tokenizer.pad_tok,
            tokenizer.unk_tok,
            tokenizer.dim_tok,
            tokenizer.delimiter_tok,
        )
    ]

    # Grammar FSM: seed its state from the priming context so the first generated
    # token already obeys the note/pedal grammar. None => structural-only masking.
    grammar, fsm_cls = _build_grammar_cached(tokenizer)
    fsm = None
    if grammar is not None and fsm_cls is not None:
        try:
            fsm = fsm_cls(grammar)
            fsm.replay([int(t) for t in enc_seq[0, :idx].tolist()])
        except Exception:  # noqa: BLE001 — degrade to structural masking
            fsm = None

    while (
        (not control_sentinel.is_set())
        and idx < MAX_SEQ_LEN
        and (deadline_ms is None or get_epoch_time_ms() < deadline_ms)
    ):
        decode_one_start_time_s = time.time()
        prev_tok_id = enc_seq[0, idx - 1]
        prev_tok = tokenizer.id_to_tok[prev_tok_id.item()]

        logits = decode_one(
            model,
            idxs=mx.array([[prev_tok_id]], dtype=mx.int32),
            input_pos=mx.array([idx - 1], dtype=mx.int32),
        )

        logger.debug(
            f"Sampled logits for positions {idx} by inserting {prev_tok} at position {idx - 1}"
        )

        logits[:, tokenizer.tok_to_id[tokenizer.ped_off_tok]] += 3  # Manual adj
        logits[:, structural_mask_ids] = float("-inf")

        logits[:, dur_mask_ids] = float("-inf")
        if last_tok_is_pedal is True:
            logits[:, dur_ids] = float("-inf")

        if is_ending is False:
            logits[:, tokenizer.tok_to_id[tokenizer.eos_tok]] = float("-inf")

        # Grammar mask: restrict to the categories valid in the current FSM state
        # (note/pedal/time/eos at top; onset after a note/pedal; dur after onset).
        # Additive -inf, layered on top of the structural masks above.
        if fsm is not None:
            logits = logits + fsm.neg_mask()[None, :]

        if temperature > 0.0:
            next_token_ids = sample_min_p(logits / temperature, min_p).flatten()
        else:
            next_token_ids = mx.argmax(logits, axis=-1).flatten()

        enc_seq[:, idx] = next_token_ids
        if fsm is not None:
            fsm.advance(int(next_token_ids[0].item()))
        next_token = tokenizer.id_to_tok[next_token_ids[0].item()]
        logger.debug(
            f"({(time.time() - decode_one_start_time_s) * 1000:.2f}ms) {idx}: {next_token}"
        )

        if next_token in {tokenizer.ped_on_tok, tokenizer.ped_off_tok}:
            last_tok_is_pedal = True
        elif isinstance(next_token, tuple) and next_token[0] == "piano":
            last_tok_is_pedal = False

        if next_token == tokenizer.eos_tok:
            logger.info("EOS token produced")
            generated_tokens_queue.put(next_token)
            return
        else:
            generated_tokens_queue.put(next_token)
            idx += 1

    logger.info(f"Finished generating: {idx}")
    generated_tokens_queue.put(None)


def generate_tokens(
    priming_seq: list,
    tokenizer: AbsTokenizer,
    model: TransformerLM,
    prev_context: list[int],
    control_sentinel: threading.Event,
    generated_tokens_queue: queue.Queue,
    num_preceding_active_pitches: int,
    first_on_msg_epoch_ms: int,
    temperature: float = 0.98,
    min_p: float = 0.03,
    is_ending: bool = False,
    max_gen_ms: int | None = None,
    catchup_discard: bool = False,
):
    logger = get_logger("GENERATE")

    generate_start_s = time.time()
    priming_seq_len = len(priming_seq)

    start_idx = max(
        2, priming_seq_len - 3 * (num_preceding_active_pitches + 2) - 1
    )
    enc_seq = mx.array(
        [
            tokenizer.encode(
                priming_seq
                + [tokenizer.pad_tok] * (MAX_SEQ_LEN - len(priming_seq))
            )
        ],
        dtype=mx.int32,
    )

    logger.debug(f"Priming sequence {priming_seq}")
    logger.info(f"Priming sequence length: {priming_seq_len}")

    logger.info(f"Prefilling up to (and including) position: {start_idx - 1}")

    prefill_start_s = time.time()
    chunked_prefill(
        model=model,
        tokenizer=tokenizer,
        prev_context=prev_context,
        curr_context=enc_seq[0, :start_idx].tolist(),
        full=True,
    )

    logger.info(
        f"Prefill took {(time.time() - prefill_start_s) * 1000:.2f} milliseconds"
    )
    logger.info(
        f"Starting duration recalculation from position: {start_idx - 1}"
    )

    recalculate_dur_start_s = time.time()
    enc_seq, priming_seq, next_token_logits = recalc_dur_tokens_chunked(
        model=model,
        priming_seq=priming_seq,
        enc_seq=enc_seq,
        tokenizer=tokenizer,
        start_idx=start_idx,
    )

    logger.info(
        f"Recalculating durations took {(time.time() - recalculate_dur_start_s) * 1000:.2f} milliseconds"
    )

    decode_first_s = time.time()
    enc_seq, idx = decode_first_tokens(
        model=model,
        first_token_logits=next_token_logits,
        enc_seq=enc_seq,
        priming_seq=priming_seq,
        tokenizer=tokenizer,
        generated_tokens_queue=generated_tokens_queue,
        first_on_msg_epoch_ms=first_on_msg_epoch_ms,
        catchup_discard=catchup_discard,
    )

    logger.info(
        f"Decode first two tokens took {(time.time() - decode_first_s) * 1000:.2f} milliseconds"
    )
    logger.info(
        f"Time to first token took {(time.time() - generate_start_s) * 1000:.2f} milliseconds"
    )

    decode_tokens(
        model=model,
        enc_seq=enc_seq,
        tokenizer=tokenizer,
        control_sentinel=control_sentinel,
        generated_tokens_queue=generated_tokens_queue,
        idx=idx,
        temperature=temperature,
        min_p=min_p,
        is_ending=is_ending,
        max_gen_ms=max_gen_ms,
    )


def _adjust_previous_off_time(
    pitch_to_prev_msg: dict,
    key: str | int,
    new_on_send_time: int,
    min_delta_ms: int,
    logger: logging.Logger,
):
    prev_on, prev_off = pitch_to_prev_msg.get(key, (None, None))

    if prev_on is not None and prev_off is not None and min_delta_ms > 0:
        adj_send_off_time = max(
            min(
                prev_off["send_epoch_time_ms"],
                new_on_send_time - min_delta_ms,
            ),
            prev_on[
                "send_epoch_time_ms"
            ],  #  Don't move prev_off before prev_on
        )
        if adj_send_off_time != prev_off["send_epoch_time_ms"]:
            logger.debug(f"Adjusting {prev_off}: t={adj_send_off_time}")
            prev_off["send_epoch_time_ms"] = adj_send_off_time
            prev_off["adjusted"] = True


# TODO: Verify that only ON -> OFF sequences are possible in tokenizer
def _decode_pedal_double(
    note_buffer: list,
    first_on_msg_epoch_ms: int,
    num_time_toks: int,
    pitch_to_prev_msg: dict,
    outbound_midi_msg_queue: queue.Queue,
    logger: logging.Logger,
    tokenizer: AbsTokenizer,
):
    pedal_tok, onset_tok = note_buffer
    velocity = 127 if pedal_tok == tokenizer.ped_on_tok else 0
    _, onset = onset_tok

    onset_epoch_ms = first_on_msg_epoch_ms + (num_time_toks * 5000) + onset
    send_onset_epoch_ms = onset_epoch_ms - BASE_OUTPUT_LATENCY_MS
    pedal_msg = {
        "pitch": "pedal",
        "vel": velocity,
        "epoch_time_ms": onset_epoch_ms,
        "send_epoch_time_ms": send_onset_epoch_ms,
        "uuid": "pedal",  # All pedals have the same id
    }

    if pedal_tok == tokenizer.ped_on_tok:
        _adjust_previous_off_time(
            pitch_to_prev_msg=pitch_to_prev_msg,
            key="pedal",
            new_on_send_time=send_onset_epoch_ms,
            min_delta_ms=MIN_PEDAL_DELTA_MS,
            logger=logger,
        )
        pitch_to_prev_msg["pedal"] = (pedal_msg, None)

    elif pedal_tok == tokenizer.ped_off_tok:
        prev_on, _ = pitch_to_prev_msg.get("pedal", (None, None))
        pitch_to_prev_msg["pedal"] = (prev_on, pedal_msg)

    outbound_midi_msg_queue.put(pedal_msg)
    logger.debug(f"Put message: {pedal_msg}")
    logger.debug(f"Ahead by {onset_epoch_ms - get_epoch_time_ms()}ms")

    return onset_epoch_ms


def _decode_note_triple(
    note_buffer: list,
    first_on_msg_epoch_ms: int,
    num_time_toks: int,
    pitch_to_prev_msg: dict,
    outbound_midi_msg_queue: queue.Queue,
    logger: logging.Logger,
):
    note_tok, onset_tok, dur_tok = note_buffer
    _, pitch, vel = note_tok
    _, onset = onset_tok
    _, dur = dur_tok

    _uuid = uuid.uuid4()
    onset_epoch_ms = first_on_msg_epoch_ms + (num_time_toks * 5000) + onset
    offset_epoch_ms = onset_epoch_ms + dur
    send_onset_epoch_ms = onset_epoch_ms - _get_input_latency_ms(vel)
    send_offset_epoch_ms = offset_epoch_ms - BASE_OUTPUT_LATENCY_MS

    on_msg = {
        "pitch": pitch,
        "vel": vel,
        "epoch_time_ms": onset_epoch_ms,
        "send_epoch_time_ms": send_onset_epoch_ms,
        "uuid": _uuid,
    }
    off_msg = {
        "pitch": pitch,
        "vel": 0,
        "epoch_time_ms": offset_epoch_ms,
        "send_epoch_time_ms": send_offset_epoch_ms,
        "uuid": _uuid,
    }

    _adjust_previous_off_time(
        pitch_to_prev_msg=pitch_to_prev_msg,
        key=pitch,
        new_on_send_time=send_onset_epoch_ms,
        min_delta_ms=MIN_NOTE_DELTA_MS,
        logger=logger,
    )

    pitch_to_prev_msg[pitch] = (on_msg, off_msg)

    outbound_midi_msg_queue.put(on_msg)
    outbound_midi_msg_queue.put(off_msg)
    logger.debug(f"Put message: {on_msg}")
    logger.debug(f"Put message: {off_msg}")
    logger.debug(f"Ahead by {onset_epoch_ms - get_epoch_time_ms()}ms")

    return offset_epoch_ms


# TODO: Refactor this method to prettify it
def decode_tokens_to_midi(
    generated_tokens_queue: queue.Queue,
    outbound_midi_msg_queue: queue.Queue,
    tokenizer: AbsTokenizer,
    first_on_msg_epoch_ms: int,
    priming_seq_last_onset_ms: int,
):
    logger = get_logger("DECODE")

    # Normally the context's last onset is in the past. In duet mode the model
    # generates ahead of wall-clock, so a burst can briefly start while the
    # previous context still extends into the future. run_duet catches wall-clock
    # up before each burst to keep this rare; downgrade the old hard assert to a
    # warning so an edge case can't kill this thread (which would hang stream_midi).
    if (
        first_on_msg_epoch_ms + priming_seq_last_onset_ms
        >= get_epoch_time_ms() + HARDWARE_INPUT_LATENCY_MS
    ):
        logger.warning(
            "Context last onset is ahead of wall-clock by "
            f"{first_on_msg_epoch_ms + priming_seq_last_onset_ms - get_epoch_time_ms()}ms; "
            "proceeding (notes will be scheduled slightly ahead)."
        )

    logger.info(f"Priming sequence last onset: {priming_seq_last_onset_ms}")
    logger.info(
        f"Total time elapsed since first onset: {get_epoch_time_ms() - first_on_msg_epoch_ms}"
    )

    pitch_to_prev_msg = {}
    note_buffer = []
    num_time_toks = priming_seq_last_onset_ms // 5000
    # Running offset of the last decoded note; also the fallback time for the
    # end marker if generation stops before any note is decoded.
    offset_epoch_ms = first_on_msg_epoch_ms + priming_seq_last_onset_ms

    while True:
        while True:
            tok = generated_tokens_queue.get()
            if tok is tokenizer.eos_tok:
                # pitch=-1 is interpreted as the end message by stream_midi
                _uuid = uuid.uuid4()
                end_msg = {
                    "pitch": -1,
                    "vel": -1,
                    "epoch_time_ms": offset_epoch_ms + 100,
                    "send_epoch_time_ms": offset_epoch_ms + 100,
                    "uuid": _uuid,
                }
                outbound_midi_msg_queue.put(end_msg)
                logger.info(f"Seen exit signal: EOS token")
                logger.debug(f"Put message: {end_msg}")
                return

            elif tok is None:
                # Deadline / sequence-limit end (e.g. duet mode). Emit the same
                # pitch=-1 end marker the EOS path uses, otherwise stream_midi
                # loops forever waiting for a control signal that never arrives.
                _uuid = uuid.uuid4()
                end_msg = {
                    "pitch": -1,
                    "vel": -1,
                    "epoch_time_ms": offset_epoch_ms + 100,
                    "send_epoch_time_ms": offset_epoch_ms + 100,
                    "uuid": _uuid,
                }
                outbound_midi_msg_queue.put(end_msg)
                logger.info(f"Seen exit signal: Sentinel")
                return

            logger.debug(f"Seen token: {tok}")
            note_buffer.append(tok)

            if isinstance(tok, tuple) and tok[0] == "dur":
                msg_type = "note"
                break
            elif (
                isinstance(tok, tuple)
                and tok[0] == "onset"
                and len(note_buffer) >= 2  # guard: onset can be the 1st token
                and note_buffer[-2]
                in {tokenizer.ped_on_tok, tokenizer.ped_off_tok}
            ):
                msg_type = "pedal"
                break

        while note_buffer and note_buffer[0] == tokenizer.time_tok:
            logger.debug("Popping time_tok")
            num_time_toks += 1
            note_buffer.pop(0)

        # Defensive: the model can emit a malformed group -- consecutive pitch
        # tokens, a missing onset, or (at extreme sampling) a stray structural
        # token. Require a real note [piano, onset, dur] or pedal [PED, onset]
        # and skip anything else, rather than crash the decoder thread.
        valid_note = (
            msg_type == "note"
            and len(note_buffer) == 3
            and isinstance(note_buffer[0], tuple) and note_buffer[0][0] == "piano"
            and isinstance(note_buffer[1], tuple) and note_buffer[1][0] == "onset"
            and isinstance(note_buffer[2], tuple) and note_buffer[2][0] == "dur"
        )
        valid_pedal = (
            msg_type == "pedal"
            and len(note_buffer) == 2
            and note_buffer[0] in {tokenizer.ped_on_tok, tokenizer.ped_off_tok}
            and isinstance(note_buffer[1], tuple) and note_buffer[1][0] == "onset"
        )
        if not (valid_note or valid_pedal):
            logger.warning(f"Skipping malformed token group: {note_buffer}")
            note_buffer = []
            continue

        logger.debug(f"Decoded note: {note_buffer}")

        try:
            if msg_type == "note":
                offset_epoch_ms = _decode_note_triple(
                    note_buffer=note_buffer,
                    first_on_msg_epoch_ms=first_on_msg_epoch_ms,
                    num_time_toks=num_time_toks,
                    pitch_to_prev_msg=pitch_to_prev_msg,
                    outbound_midi_msg_queue=outbound_midi_msg_queue,
                    logger=logger,
                )
            else:  # pedal
                offset_epoch_ms = _decode_pedal_double(
                    note_buffer=note_buffer,
                    first_on_msg_epoch_ms=first_on_msg_epoch_ms,
                    num_time_toks=num_time_toks,
                    pitch_to_prev_msg=pitch_to_prev_msg,
                    outbound_midi_msg_queue=outbound_midi_msg_queue,
                    logger=logger,
                    tokenizer=tokenizer,
                )
        except Exception as e:
            logger.warning(f"Skipping un-decodable token group {note_buffer}: {e}")

        note_buffer = []


def _create_mido_message(
    msg_dict: dict,
    channel: int,
    time_delta_ms: int,
) -> mido.Message:
    if msg_dict["pitch"] == "pedal":
        return mido.Message(
            "control_change",
            control=64,
            value=msg_dict["vel"],
            channel=channel,
            time=time_delta_ms,
        )
    else:
        # note-on or note-off
        return mido.Message(
            "note_on",
            note=msg_dict["pitch"],
            velocity=msg_dict["vel"],
            channel=channel,
            time=time_delta_ms,
        )


def stream_midi(
    inbound_midi_msg_queue: queue.Queue,
    msgs: list[mido.Message],
    last_channel_msg_epoch_time_ms: float,
    midi_output_port: str,
    control_sentinel: threading.Event,
    midi_stream_channel: int,
    results_queue: queue.Queue,
):
    logger = get_logger("STREAM")
    logger.info(f"Sending generated messages on port: '{midi_output_port}'")
    global CURRENT_STREAM_LAG_MS
    CURRENT_STREAM_LAG_MS = 0.0
    # Phrase mode: start the timeline already shifted by the lead, so notes play
    # PHRASE_LEAD_MS after they're scheduled -> the decode runs ahead and the
    # response plays back smoothly. (Live mode keeps this 0.)
    lead_ms = float(PHRASE_LEAD_MS)

    active_pitch_uuid = {}
    pending_msgs = []
    msgs_to_archive = []
    # Per-note bookkeeping for duration preservation. When the decode falls
    # behind, a note's onset AND offset both sit in the past, so a naive
    # scheduler fires the note-off immediately after the note-on and the note
    # collapses to a click — turning a held chord into a machine-gun arpeggio.
    # We remember when each note-on was *actually* played and hold its note-off
    # until the note has sounded for its intended duration.
    on_play_wall_ms = {}   # uuid -> wall time the note-on was actually sent
    on_epoch_ms = {}       # uuid -> the note-on's intended epoch_time_ms
    # When the decode falls behind real-time, we don't DROP notes (that's just
    # silence) — we stretch the playback timeline by the accumulated lag so the
    # stream keeps playing at the decode's pace with relative rhythm intact.
    # Aria keeps up so this normally stays ~lead; a slower decode grows it
    # gradually.
    schedule_offset_ms = lead_ms

    with open_output(midi_output_port) as midi_out:
        while not control_sentinel.is_set():
            while not inbound_midi_msg_queue.empty():
                try:
                    msg = inbound_midi_msg_queue.get_nowait()
                    if msg:
                        pending_msgs.append(msg)
                except queue.Empty:
                    break

            pending_msgs.sort(key=lambda m: (m["send_epoch_time_ms"], m["vel"]))

            # Drain by index (not pop-front + break): a note-off may be held
            # into the future to preserve its note's duration, and that must NOT
            # block the note-ons sorted behind it (which keep the line flowing).
            i = 0
            while i < len(pending_msgs):
                curr_epoch_time_ms = get_epoch_time_ms()
                msg = pending_msgs[i]

                eff_send_ms = msg["send_epoch_time_ms"] + schedule_offset_ms

                # Duration preservation. A note-off's absolute schedule sits in
                # the past once the decode is behind, which would fire it right
                # after its note-on (a click). Instead, hold the off until its
                # own note-on has actually sounded for the intended duration.
                is_note_off = msg["vel"] == 0 and msg["pitch"] != "pedal"
                if (
                    is_note_off
                    and not msg.get("adjusted", False)
                    and msg["uuid"] in on_play_wall_ms
                ):
                    intended_dur_ms = msg["epoch_time_ms"] - on_epoch_ms.get(
                        msg["uuid"], msg["epoch_time_ms"]
                    )
                    intended_dur_ms = max(
                        MIN_NOTE_LENGTH_MS, min(intended_dur_ms, MAX_NOTE_HOLD_MS)
                    )
                    hold_until_ms = on_play_wall_ms[msg["uuid"]] + intended_dur_ms
                    eff_send_ms = max(eff_send_ms, hold_until_ms)

                if eff_send_ms > curr_epoch_time_ms:
                    # Not due yet. Skip it and keep draining the rest -- a held
                    # note-off must not stall the note-ons after it.
                    i += 1
                    continue

                late_ms = curr_epoch_time_ms - eff_send_ms
                if late_ms > MAX_STREAM_DELAY_MS and msg["vel"] > 0:
                    # Decode fell behind real-time: stretch the whole timeline by
                    # the lag (instead of dropping the note -> silence). All
                    # remaining notes shift by the same amount, so the music keeps
                    # playing at the decode pace with its relative rhythm intact.
                    # Driven by onsets only -- held note-offs are intentionally in
                    # the future and must not inflate the reported lag.
                    schedule_offset_ms += late_ms
                    # Report lag BEYOND the intentional phrase lead (so a buffered
                    # phrase reads as in-sync until the decode falls behind it).
                    CURRENT_STREAM_LAG_MS = max(0.0, schedule_offset_ms - lead_ms)
                    logger.info(
                        f"Decode behind real-time; stretched playback "
                        f"(+{late_ms:.0f}ms, total {schedule_offset_ms:.0f}ms) "
                        "instead of dropping notes"
                    )

                logger.debug(f"Processing: {msg}")

                # End signal
                if msg["pitch"] == -1:
                    control_sentinel.set()
                    break

                should_send = False
                should_archive = False
                if msg["vel"] > 0:  # note-on or pedal-on
                    active_pitch_uuid[msg["pitch"]] = msg["uuid"]
                    should_send = True
                    should_archive = True
                    if msg["pitch"] != "pedal":
                        on_play_wall_ms[msg["uuid"]] = curr_epoch_time_ms
                        on_epoch_ms[msg["uuid"]] = msg["epoch_time_ms"]
                else:  # note-off or pedal-off (vel == 0)
                    if msg.get("adjusted", False):
                        should_send = True
                        should_archive = msg["pitch"] == "pedal"
                    elif active_pitch_uuid.get(msg["pitch"]) == msg["uuid"]:
                        should_send = True
                        should_archive = True
                        active_pitch_uuid.pop(msg["pitch"], None)
                    on_play_wall_ms.pop(msg["uuid"], None)
                    on_epoch_ms.pop(msg["uuid"], None)

                if should_send:
                    mido_msg = _create_mido_message(
                        msg_dict=msg, channel=0, time_delta_ms=0
                    )
                    midi_out.send(mido_msg)
                    logger.info(f"Sent message: {mido_msg}")

                if should_archive:
                    msgs_to_archive.append(msg)

                pending_msgs.pop(i)

            if control_sentinel.is_set():
                break

            time.sleep(0.005)

        last_archive_time_ms = last_channel_msg_epoch_time_ms
        msgs_to_archive.sort(key=lambda m: (m["epoch_time_ms"], m["vel"]))

        for msg in msgs_to_archive:
            time_delta_ms = round(msg["epoch_time_ms"] - last_archive_time_ms)
            mido_msg = _create_mido_message(
                msg_dict=msg,
                channel=midi_stream_channel,
                time_delta_ms=time_delta_ms,
            )
            msgs.append(mido_msg)
            last_archive_time_ms = msg["epoch_time_ms"]

        logger.info("Sending final note-off messages for cleanup.")
        remaining_off_msgs = [
            msg
            for msg in pending_msgs
            if msg["vel"] == 0
            and msg["pitch"] != "pedal"
            and active_pitch_uuid.get(msg["pitch"]) == msg["uuid"]
        ]
        remaining_off_msgs.sort(key=lambda m: m["epoch_time_ms"])

        for msg in remaining_off_msgs:
            mido_msg = _create_mido_message(
                msg_dict=msg, channel=0, time_delta_ms=0
            )
            midi_out.send(mido_msg)

            time_delta_ms = round(msg["epoch_time_ms"] - last_archive_time_ms)
            archived_msg = _create_mido_message(
                msg_dict=msg,
                channel=midi_stream_channel,
                time_delta_ms=time_delta_ms,
            )
            msgs.append(archived_msg)
            last_archive_time_ms = msg["epoch_time_ms"]

        midi_out.send(
            mido.Message(
                "control_change", control=64, value=0, channel=0, time=0
            )
        )

    # Also report the epoch of the last scheduled note so duet mode can let
    # wall-clock catch up to the model's look-ahead before the next burst.
    results_queue.put((msgs, last_archive_time_ms))


def stream_msgs(
    model: TransformerLM,
    tokenizer: AbsTokenizer,
    msgs: list[mido.Message],
    prev_context: list[int],
    midi_output_port: str,
    first_on_msg_epoch_ms: int,
    control_sentinel: threading.Event,
    temperature: float,
    min_p: float,
    num_preceding_active_pitches: int,
    midi_stream_channel: int,
    is_ending: bool = False,
    max_gen_ms: int | None = None,
    timing_out: dict | None = None,
):

    logger = get_logger("STREAM")
    midi = convert_msgs_to_midi(msgs=msgs)
    midi_dict = MidiDict(**midi_to_dict(midi))
    midi_dict.remove_redundant_pedals()
    priming_seq = tokenizer.tokenize(midi_dict=midi_dict, add_dim_tok=False)
    priming_seq = priming_seq[: priming_seq.index(tokenizer.eos_tok)]

    if priming_seq[-2] == tokenizer.ped_off_tok:
        # Final pedal-off is needed for tokenizer, but unneeded in tokenized sequence
        logger.info("Removing final pedal_off from tokenized sequence")
        priming_seq = priming_seq[:-2]

    priming_seq.append(tokenizer.delimiter_tok)

    if is_ending is True:
        raise NotImplementedError(
            "I've removed this functionality so this should never trigger"
        )
        priming_seq.append(tokenizer.dim_tok)

    # Seamless turn-switch: freeze the clock so the model continues on-grid from
    # just after the last captured note instead of snapping its first note to
    # wall-clock. Re-anchoring first_on here keeps generation timing, output
    # scheduling, and the archive baseline (all below) mutually consistent.
    # Gated to turn-taking (max_gen_ms is None); duet manages its own timing.
    # Degrades gracefully: if the freeze latency is under-estimated the model
    # simply catches up toward now (never schedules in the past), so no notes
    # are dropped.
    # Turn-switch timing modes (turn-taking only; duet manages its own timing).
    catchup_discard = False
    if max_gen_ms is None and not is_ending:
        if TURN_SWITCH_MODE == "freeze":
            priming_last_onset_ms = tokenizer.calc_length_ms(
                priming_seq, onset=True
            )
            first_on_msg_epoch_ms = (
                get_epoch_time_ms()
                - priming_last_onset_ms
                + TURN_FREEZE_LATENCY_MS
            )
            logger.info(
                f"Turn-switch [freeze]: froze clock "
                f"(latency={TURN_FREEZE_LATENCY_MS}ms), re-anchored first_on"
            )
        elif TURN_SWITCH_MODE == "catchup":
            # Keep the original anchor; tell decode_first_tokens to NOT insert
            # catch-up time tokens or mask past onsets, so the model continues
            # from the last note. Notes whose scheduled time is already in the
            # past are dropped downstream by stream_midi's stale-skip, so the
            # first audible note lands on the original grid at wall-clock now.
            catchup_discard = True
            logger.info(
                "Turn-switch [catchup]: continue through the pause, discard "
                "past notes, join on the original grid"
            )

    generated_tokens_queue = queue.Queue()
    midi_messages_queue = queue.Queue()

    generate_tokens_thread = threading.Thread(
        target=generate_tokens,
        kwargs={
            "priming_seq": priming_seq,
            "tokenizer": tokenizer,
            "model": model,
            "prev_context": prev_context,
            "control_sentinel": control_sentinel,
            "generated_tokens_queue": generated_tokens_queue,
            "temperature": temperature,
            "min_p": min_p,
            "num_preceding_active_pitches": num_preceding_active_pitches,
            "first_on_msg_epoch_ms": first_on_msg_epoch_ms,
            "is_ending": is_ending,
            "max_gen_ms": max_gen_ms,
            "catchup_discard": catchup_discard,
        },
    )
    generate_tokens_thread.start()

    decode_tokens_to_midi_thread = threading.Thread(
        target=decode_tokens_to_midi,
        kwargs={
            "generated_tokens_queue": generated_tokens_queue,
            "outbound_midi_msg_queue": midi_messages_queue,
            "tokenizer": tokenizer,
            "first_on_msg_epoch_ms": first_on_msg_epoch_ms,
            "priming_seq_last_onset_ms": tokenizer.calc_length_ms(
                priming_seq, onset=True
            ),
        },
    )
    decode_tokens_to_midi_thread.start()

    # If ending==True then previous MIDI message on midi_stream_channel occurs
    # at first_on_msg_epoch_ms.
    prev_channel_msg_epoch_time_ms = (
        first_on_msg_epoch_ms
        + tokenizer.calc_length_ms(priming_seq, onset=False)
        if is_ending is False
        else first_on_msg_epoch_ms
    )

    stream_midi_results_queue = queue.Queue()
    stream_midi_thread = threading.Thread(
        target=stream_midi,
        kwargs={
            "inbound_midi_msg_queue": midi_messages_queue,
            "msgs": msgs,
            "last_channel_msg_epoch_time_ms": prev_channel_msg_epoch_time_ms,
            "midi_output_port": midi_output_port,
            "control_sentinel": control_sentinel,
            "midi_stream_channel": midi_stream_channel,
            "results_queue": stream_midi_results_queue,
        },
    )
    stream_midi_thread.start()

    generate_tokens_thread.join()
    decode_tokens_to_midi_thread.join()
    stream_midi_thread.join()
    msgs, last_scheduled_epoch_ms = stream_midi_results_queue.get()
    if timing_out is not None:
        timing_out["last_epoch_ms"] = last_scheduled_epoch_ms

    return msgs


def convert_msgs_to_midi(msgs: list[mido.Message]):
    channel_to_track = {
        chan: mido.MidiTrack()
        for chan in list(set([msg.channel for msg in msgs]))
    }

    for msg in msgs:
        channel_to_track[msg.channel].append(msg)

    # Workaround for possibility that track_0 start time != first_on_msg_epoch_ms
    for msg in channel_to_track[0]:
        if msg.type == "note_on" and msg.velocity > 0:
            msg.time = 0
            break
        else:
            msg.time = 0

    mid = mido.MidiFile(type=1)
    mid.ticks_per_beat = 500

    for channel, track in channel_to_track.items():
        track.insert(0, mido.MetaMessage("set_tempo", tempo=500000, time=0))
        track.insert(
            0,
            mido.Message("program_change", program=0, channel=channel, time=0),
        )
        mid.tracks.append(track)

    return mid


def _find_divergence(
    prev_context: list,
    curr_context: list,
    logger: logging.Logger,
    tokenizer: AbsTokenizer,
):
    agreement_index = 0
    for prev_val, curr_val in zip(prev_context, curr_context):
        if prev_val == curr_val:
            agreement_index += 1
        else:
            logger.info(
                f"Found divergence at idx {agreement_index}: {tokenizer.id_to_tok[curr_val]}, {tokenizer.id_to_tok[prev_val]}"
            )
            break

    return agreement_index, curr_context[agreement_index:]


def chunked_prefill(
    model: TransformerLM,
    tokenizer: AbsTokenizer,
    prev_context: list,
    curr_context: list,
    full: bool = False,
):

    assert isinstance(curr_context[0], int)
    assert tokenizer.pad_id not in prev_context
    assert tokenizer.pad_id not in curr_context

    logger = get_logger("PREFILL")

    while True:
        prefill_idx, prefill_toks = _find_divergence(
            prev_context,
            curr_context,
            logger=logger,
            tokenizer=tokenizer,
        )
        num_prefill_toks = len(prefill_toks)
        logger.debug(f"Tokens to prefill: {len(prefill_toks)}")

        if num_prefill_toks > PREFILL_CHUNK_SIZE_L:
            logger.debug(
                f"Prefilling {PREFILL_CHUNK_SIZE_L} tokens from idx={prefill_idx}"
            )
            mx.eval(
                prefill(
                    model,
                    idxs=mx.array(
                        [prefill_toks[:PREFILL_CHUNK_SIZE_L]],
                        dtype=mx.int32,
                    ),
                    input_pos=mx.arange(
                        prefill_idx,
                        prefill_idx + PREFILL_CHUNK_SIZE_L,
                        dtype=mx.int32,
                    ),
                )
            )
            prev_context = curr_context[: prefill_idx + PREFILL_CHUNK_SIZE_L]

        elif num_prefill_toks > PREFILL_CHUNK_SIZE:
            logger.debug(
                f"Prefilling {PREFILL_CHUNK_SIZE} tokens from idx={prefill_idx}"
            )
            mx.eval(
                prefill(
                    model,
                    idxs=mx.array(
                        [prefill_toks[:PREFILL_CHUNK_SIZE]],
                        dtype=mx.int32,
                    ),
                    input_pos=mx.arange(
                        prefill_idx,
                        prefill_idx + PREFILL_CHUNK_SIZE,
                        dtype=mx.int32,
                    ),
                )
            )
            prev_context = curr_context[: prefill_idx + PREFILL_CHUNK_SIZE]

        elif num_prefill_toks > 0 and full is True:
            logger.debug(
                f"Prefilling (force) {num_prefill_toks} tokens from idx={prefill_idx}"
            )
            prefill_toks += (PREFILL_CHUNK_SIZE - len(prefill_toks)) * [
                tokenizer.pad_id
            ]
            mx.eval(
                prefill(
                    model,
                    idxs=mx.array([prefill_toks], dtype=mx.int32),
                    input_pos=mx.arange(
                        prefill_idx,
                        prefill_idx + PREFILL_CHUNK_SIZE,
                        dtype=mx.int32,
                    ),
                )
            )
            prev_context = curr_context
            break
        else:
            break

    logger.info(
        f"KV stored up to idx={max(0, len(prev_context) - 1)} (curr_context_len={len(curr_context)})"
    )

    return prev_context


def continuous_prefill(
    model: TransformerLM,
    msgs: list,
    received_messages_queue: queue.Queue,
    prev_context: list[int],
):
    tokenizer = AbsTokenizer(config_path=TOKENIZER_CONFIG_PATH)
    logger = get_logger("PREFILL")
    msg_cnt = 0
    seen_sentinel = False

    while seen_sentinel is False:
        while seen_sentinel is False:
            try:
                msg = received_messages_queue.get_nowait()
            except queue.Empty:
                break
            else:
                if msg is None:
                    logger.info("Seen sentinel in message received messages")
                    seen_sentinel = True
                else:
                    msgs.append(msg)
                    msg_cnt += 1

        if msg_cnt >= 10:
            midi = convert_msgs_to_midi(msgs=msgs)
            midi_dict = MidiDict(**midi_to_dict(midi))
            midi_dict.remove_redundant_pedals()

            if len(midi_dict.note_msgs) > 0:
                curr_context = tokenizer.encode(
                    tokenizer.tokenize(midi_dict, add_dim_tok=False)
                )
                prev_context = chunked_prefill(
                    model=model,
                    tokenizer=tokenizer,
                    prev_context=prev_context,
                    curr_context=curr_context,
                    full=False,
                )

            msg_cnt = 0
        else:
            time.sleep(0.01)

    return msgs, prev_context


def capture_and_update_kv(
    model: TransformerLM,
    msgs: list,
    prev_context: list,
    control_sentinel: threading.Event,
    reset_sentinel: threading.Event,
    wait_for_close: bool,
    midi_performance_queue: queue.Queue,
    midi_capture_channel: int,
    first_msg_epoch_time_ms: int | None = None,
    flush_pending: bool = True,
):
    received_messages_queue = queue.Queue()
    results_queue = queue.Queue()
    capture_midi_thread = threading.Thread(
        target=capture_midi_input,
        kwargs={
            "midi_performance_queue": midi_performance_queue,
            "control_sentinel": control_sentinel,
            "reset_sentinel": reset_sentinel,
            "received_messages_queue": received_messages_queue,
            "midi_capture_channel": midi_capture_channel,
            "first_msg_epoch_time_ms": first_msg_epoch_time_ms,
            "results_queue": results_queue,
            "wait_for_close": wait_for_close,
            "flush_pending": flush_pending,
        },
    )
    capture_midi_thread.start()

    msgs, prev_context = continuous_prefill(
        model=model,
        msgs=msgs,
        received_messages_queue=received_messages_queue,
        prev_context=prev_context,
    )
    capture_midi_thread.join()
    first_on_msg_epoch_ms, num_active_pitches = results_queue.get()

    return msgs, prev_context, first_on_msg_epoch_ms, num_active_pitches


def capture_midi_input(
    midi_performance_queue: queue.Queue,
    control_sentinel: threading.Event,
    reset_sentinel: threading.Event,
    received_messages_queue: queue.Queue,
    midi_capture_channel: int,
    results_queue: queue.Queue,
    first_msg_epoch_time_ms: int | None = None,
    wait_for_close: bool = False,
    flush_pending: bool = True,
):
    logger = get_logger("CAPTURE")

    # Timeline baseline: epoch of the previous message ACTUALLY emitted to the
    # context. Updated only on emit (in emit_to_context), so dropped pedal values
    # between events never steal elapsed time from the context timeline.
    last_emit_epoch_ms = first_msg_epoch_time_ms

    def emit_to_context(m, epoch_ms):
        # Log (at INFO) and enqueue EXACTLY the messages that populate the
        # model's context. The time delta is measured from the previous EMITTED
        # message, not the previous raw input -- so filtered/deduped pedal values
        # (and the gaps around them) don't compress the timeline. Raw input is
        # logged at DEBUG.
        nonlocal last_emit_epoch_ms
        m.time = 0 if last_emit_epoch_ms is None else (epoch_ms - last_emit_epoch_ms)
        last_emit_epoch_ms = epoch_ms
        logger.info(f"[CONTEXT] {m}")
        received_messages_queue.put(m)

    first_on_msg_epoch_ms = None
    pedal_down = False
    pitches_held_down = set()
    pitches_sustained_by_pedal = set()

    # A fresh start (e.g. right after a reset) has no prior context, signalled by
    # first_msg_epoch_time_ms being None. In that case begin the context at the
    # first note: ignore any leading messages (e.g. a continuously-streaming
    # sustain pedal) and the silence before the first note-on, so the recorded
    # timeline starts exactly at that note rather than at "capture start".
    waiting_for_first_note = first_msg_epoch_time_ms is None

    # In duet mode we must NOT discard notes the player performed during the
    # model's burst, so flushing is made optional.
    while flush_pending and not midi_performance_queue.empty():
        try:
            midi_performance_queue.get_nowait()
        except queue.Empty:
            break

    logger.info("Listening for input")
    logger.info("Commencing generation upon keypress or control signal")

    while True:
        epoch_time_ms = get_epoch_time_ms()
        active_notes = pitches_held_down.union(pitches_sustained_by_pedal)
        should_stop = not wait_for_close or not active_notes
        if reset_sentinel.is_set() or (
            control_sentinel.is_set() and should_stop
        ):
            break

        try:
            msg = midi_performance_queue.get(block=True, timeout=0.01)
        except queue.Empty:
            continue

        if msg.is_meta or msg.type == "program_change":
            continue

        # Drop everything before the first real event on a fresh start, so
        # leading silence (and a streaming pedal at rest) never enters the
        # context. prev epoch stays unset until then, so the first event lands
        # at time 0. A valid first event is a note-on OR a pedal-DOWN (sustain
        # pressed before the first note); a pedal-UP does not open the context.
        # NOTE: standard pedal polarity -> DOWN (pressed) is CC64 value >= 64.
        if waiting_for_first_note:
            is_first_event = (
                msg.type == "note_on" and msg.velocity > 0
            ) or (
                msg.type == "control_change"
                and msg.control == 64
                and msg.value >= 64
            )
            if is_first_event:
                waiting_for_first_note = False
                # Anchor the timeline (and wall-clock sync) to this first event,
                # whether it's the note or the opening pedal-down.
                first_on_msg_epoch_ms = epoch_time_ms - HARDWARE_INPUT_LATENCY_MS
            else:
                continue

        msg.channel = midi_capture_channel
        logger.debug(f"Raw input message: [{msg}]")

        match msg.type:
            case "note_on" if msg.velocity > 0:
                if first_on_msg_epoch_ms is None:
                    first_on_msg_epoch_ms = (
                        get_epoch_time_ms() - HARDWARE_INPUT_LATENCY_MS
                    )
                pitches_held_down.add(msg.note)
                if pedal_down:
                    pitches_sustained_by_pedal.add(msg.note)
                emit_to_context(msg, epoch_time_ms)

            case "note_off" | "note_on":
                # Note-off
                pitches_held_down.discard(msg.note)
                emit_to_context(msg, epoch_time_ms)

            case "control_change" if msg.control == 64:
                # Binary pedal with a single threshold at 64 (standard polarity):
                # DOWN (sustain on) when CC64 >= 64, UP (off) when < 64. Emit only
                # on a state change so the stream of intermediate values from a
                # continuous pedal doesn't flood the model. (Pianoteq still gets
                # the raw continuous pedal.)
                new_pedal_down = msg.value >= 64
                if new_pedal_down != pedal_down:
                    pedal_down = new_pedal_down
                    if pedal_down:
                        pitches_sustained_by_pedal.update(pitches_held_down)
                        msg.value = 127
                    else:
                        pitches_sustained_by_pedal.clear()
                        msg.value = 0
                    emit_to_context(msg, epoch_time_ms)

    active_pitches = pitches_held_down.union(pitches_sustained_by_pedal)
    num_active_pitches = len(active_pitches)
    logger.info(f"Active pitches ({num_active_pitches}): {active_pitches}")

    # Close out the capture at "now": held notes get a final note-off and the
    # pedal is released. emit_to_context sets each message's delta from the
    # previous emitted event, so the first cleanup message carries the real gap
    # since the last event and the rest are simultaneous (time 0). If nothing was
    # captured, last_emit_epoch_ms is None and these land at time 0 harmlessly.
    cleanup_epoch_ms = get_epoch_time_ms()
    for pitch in pitches_held_down:
        logger.info("[CONTEXT] (capture-end cleanup) note_off")
        emit_to_context(
            mido.Message("note_off", note=pitch, channel=midi_capture_channel),
            cleanup_epoch_ms,
        )

    logger.info("[CONTEXT] (capture-end cleanup) forced pedal-off")
    emit_to_context(
        mido.Message(
            "control_change", control=64, value=0, channel=midi_capture_channel
        ),
        cleanup_epoch_ms,
    )

    received_messages_queue.put(None)
    results_queue.put((first_on_msg_epoch_ms, num_active_pitches))


def play_midi_file(
    midi_through_port: str,
    midi_performance_queue: queue.Queue,
    midi_path: str,
    currently_generating_sentinel: threading.Event,
    reset_sentinel: threading.Event,
):
    def _send_delayed_message(_midi_performance_queue: queue.Queue, msg):
        _midi_performance_queue.put(msg)
        logger.debug(f"SENT: {msg}")

    logger = get_logger("FILE")
    logger.info(f"Playing {midi_path} on through-port '{midi_through_port}'")
    logger.info(f"Simulating input with {HARDWARE_INPUT_LATENCY_MS}ms latency")

    if BASE_OUTPUT_LATENCY_MS > 0:
        midi_dict = MidiDict.from_midi(midi_path)
        midi_dict.remove_redundant_pedals()
        midi_dict.enforce_gaps(min_gap_ms=MIN_NOTE_DELTA_MS)
        mid = midi_dict.to_midi()
    else:
        mid = mido.MidiFile(midi_path)

    time.sleep(1)
    with open_output(midi_through_port) as through_port:
        for msg in mid.play():
            if reset_sentinel.is_set():
                logger.debug("Exiting")
                return

            if currently_generating_sentinel.is_set() is False:
                through_port.send(msg)

            timer = threading.Timer(
                interval=HARDWARE_INPUT_LATENCY_MS / 1000.0,
                function=_send_delayed_message,
                args=[midi_performance_queue, msg],
            )
            timer.start()


def listen_for_keypress_control_signal(
    control_sentinel: threading.Event,
    reset_sentinel: threading.Event,
    currently_generating_sentinel: threading.Event,
    back_and_forth: bool = False,
    soft_reset_sentinel: threading.Event | None = None,
):
    logger = get_logger("KEYBOARD")
    # Without a TTY (GUI backgrounded / launched via nohup/systemd), stdin reads
    # return EOF immediately: select() reports it readable, readline() yields ""
    # which this loop treats as a takeover — firing an instant, empty-context
    # generation (crash) and spinning at 100% CPU. Idle instead; takeover still
    # works via the MIDI control CC and the GUI transport button.
    try:
        is_tty = sys.stdin is not None and sys.stdin.isatty()
    except (ValueError, OSError):
        is_tty = False
    if not is_tty:
        logger.info(
            "stdin is not a TTY; keyboard takeover disabled "
            "(use the MIDI control CC or the GUI transport button)."
        )
        reset_sentinel.wait()
        return

    logger.info(
        "Listening for keyboard input (Enter to start AI, any other key + Enter to reset)."
    )

    while not reset_sentinel.is_set():
        rlist, _, _ = select.select([sys.stdin], [], [], 0.01)

        if rlist:
            _input = sys.stdin.readline().strip()
            logger.info(f'Keypress seen "{_input}"')

            if _input == "":
                if (
                    currently_generating_sentinel.is_set()
                    and back_and_forth is False
                ):
                    logger.info("Resetting (control)")
                    reset_sentinel.set()
                control_sentinel.set()
            else:
                # A reset keypress clears context and keeps listening (soft
                # reset); it no longer tears the engine down. Hard stop is the
                # GUI Stop button / engine.stop().
                logger.info("Resetting (soft: clear context, keep listening)")
                if soft_reset_sentinel is not None:
                    soft_reset_sentinel.set()
                control_sentinel.set()

    logger.debug(
        "Exiting keypress listener because reset_sentinel was set by another thread."
    )


def _listen(
    midi_control_queue: queue.Queue,
    reset_sentinel: threading.Event,
    currently_generating_sentinel: threading.Event,
    logger: logging.Logger,
    midi_control_signal: int | None = None,
    midi_reset_control_signal: int | None = None,
):
    while not midi_control_queue.empty():
        try:
            midi_control_queue.get_nowait()
        except queue.Empty:
            break

    logger.info(
        f"Listening for takeover signal ({midi_control_signal}) and reset signal ({midi_reset_control_signal}) on control queue."
    )
    seen_note_on = False
    while not reset_sentinel.is_set():
        try:
            msg = midi_control_queue.get(block=True, timeout=0.01)
        except queue.Empty:
            continue

        if msg.type == "note_on" and msg.velocity > 0:
            seen_note_on = True

        should_return_signal = (
            seen_note_on or currently_generating_sentinel.is_set()
        )
        if (
            msg.type == "control_change"
            and msg.control == midi_control_signal
            and msg.value >= 64
            and should_return_signal
        ):
            return midi_control_signal
        elif (
            msg.type == "control_change"
            and msg.control == midi_reset_control_signal
            and msg.value >= 64
        ):
            # Reset is NOT gated by should_return_signal: clearing the context
            # needs no prior note and must work even when idle between phrases.
            return midi_reset_control_signal


def listen_for_midi_control_signal(
    midi_control_queue: queue.Queue,
    control_sentinel: threading.Event,
    reset_sentinel: threading.Event,
    currently_generating_sentinel: threading.Event,
    midi_control_signal: int | None = None,
    midi_reset_control_signal: int | None = None,
    back_and_forth: bool = False,
    soft_reset_sentinel: threading.Event | None = None,
):
    logger = get_logger("MIDI-CONTROL")

    while not reset_sentinel.is_set():
        time.sleep(1)
        signal_received = _listen(
            midi_control_queue=midi_control_queue,
            reset_sentinel=reset_sentinel,
            currently_generating_sentinel=currently_generating_sentinel,
            midi_control_signal=midi_control_signal,
            midi_reset_control_signal=midi_reset_control_signal,
            logger=logger,
        )

        if signal_received is not None:
            logger.info(f"Seen MIDI control signal ({signal_received})")

            if signal_received == midi_reset_control_signal:
                # Footswitch reset: clear context and keep listening (soft
                # reset). Does NOT stop the engine — that's the GUI Stop button.
                logger.info("Resetting (soft: clear context, keep listening)")
                if soft_reset_sentinel is not None:
                    soft_reset_sentinel.set()
                control_sentinel.set()
            elif signal_received == midi_control_signal:
                if (
                    currently_generating_sentinel.is_set()
                    and back_and_forth is False
                ):
                    logger.info("Resetting (control)")
                    reset_sentinel.set()
                control_sentinel.set()

    logger.debug("Exiting MIDI control listener")


# TODO: Debug, fix, and perhaps refactor the functionality for going back and forth
# - One idea is on resume, to wait to start the clock until the user plays.
def run(
    model: TransformerLM,
    midi_performance_queue: queue.Queue,
    midi_control_queue: queue.Queue,
    midi_through_port: str | None,
    midi_out_port: str | None,
    midi_path: str | None,
    midi_save_path: str | None,
    midi_control_signal: int,
    midi_reset_control_signal: int,
    reset_sentinel: threading.Event,
    wait_for_close: bool,
    temperature: float,
    min_p: float,
    back_and_forth: bool,
):
    logger = get_logger()
    tokenizer = AbsTokenizer(config_path=TOKENIZER_CONFIG_PATH)
    control_sentinel = threading.Event()
    currently_generating_sentinel = threading.Event()
    # Soft reset (footswitch / keyboard reset): clear the captured context and
    # any in-flight generation, then keep listening for notes. Distinct from
    # reset_sentinel, which is the HARD stop (GUI Stop / engine.stop()) that
    # tears the run loop down entirely.
    soft_reset_sentinel = threading.Event()

    if midi_through_port:
        close_notes(midi_through_port)
    if midi_out_port:
        close_notes(midi_out_port)

    if midi_path:
        play_file_thread = threading.Thread(
            target=play_midi_file,
            kwargs={
                "midi_through_port": midi_through_port,
                "midi_performance_queue": midi_performance_queue,
                "midi_path": midi_path,
                "currently_generating_sentinel": currently_generating_sentinel,
                "reset_sentinel": reset_sentinel,
            },
        )
    else:
        play_file_thread = None

    keypress_thread = threading.Thread(
        target=listen_for_keypress_control_signal,
        kwargs={
            "control_sentinel": control_sentinel,
            "reset_sentinel": reset_sentinel,
            "soft_reset_sentinel": soft_reset_sentinel,
            "currently_generating_sentinel": currently_generating_sentinel,
            "back_and_forth": back_and_forth,
        },
    )
    midi_control_thread = threading.Thread(
        target=listen_for_midi_control_signal,
        kwargs={
            "midi_control_queue": midi_control_queue,
            "control_sentinel": control_sentinel,
            "reset_sentinel": reset_sentinel,
            "soft_reset_sentinel": soft_reset_sentinel,
            "currently_generating_sentinel": currently_generating_sentinel,
            "midi_control_signal": midi_control_signal,
            "midi_reset_control_signal": midi_reset_control_signal,
            "back_and_forth": back_and_forth,
        },
    )
    keypress_thread.start()
    midi_control_thread.start()

    if play_file_thread is not None:
        play_file_thread.start()

    def _drain_queue(q):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def _fresh_listen(channel):
        # Re-establish the engine's known-good initial state: empty context, KV
        # re-prefilled from the next note onward. Identical to the very first
        # listen, so a soft reset returns the model to a clean slate.
        return capture_and_update_kv(
            model=model,
            msgs=[],
            prev_context=[],
            control_sentinel=control_sentinel,
            reset_sentinel=reset_sentinel,
            wait_for_close=wait_for_close,
            midi_performance_queue=midi_performance_queue,
            midi_capture_channel=channel,
        )

    def _apply_soft_reset():
        # Footswitch/keyboard reset: drop captured context + any hanging
        # generated notes, then start listening again. The engine stays alive.
        soft_reset_sentinel.clear()
        control_sentinel.clear()
        currently_generating_sentinel.clear()
        if midi_out_port:
            close_notes(midi_out_port)
        _drain_queue(midi_performance_queue)
        logger.info("[RESET] context cleared; listening for notes")

    msgs, prev_context, first_on_msg_epoch_ms, num_active_pitches = _fresh_listen(0)

    curr_midi_channel = 0
    while not reset_sentinel.is_set():
        # A reset that landed during the listen window: clear and re-listen,
        # don't generate from the (now empty) context.
        if soft_reset_sentinel.is_set():
            _apply_soft_reset()
            curr_midi_channel = 0
            msgs, prev_context, first_on_msg_epoch_ms, num_active_pitches = (
                _fresh_listen(0)
            )
            continue

        control_sentinel.clear()
        currently_generating_sentinel.set()
        msgs = stream_msgs(
            model=model,
            tokenizer=tokenizer,
            msgs=msgs,
            prev_context=prev_context,
            midi_output_port=midi_out_port,
            first_on_msg_epoch_ms=first_on_msg_epoch_ms,
            control_sentinel=control_sentinel,
            temperature=temperature,
            min_p=min_p,
            num_preceding_active_pitches=num_active_pitches,
            midi_stream_channel=curr_midi_channel,
            is_ending=False,
        )

        if midi_save_path and not soft_reset_sentinel.is_set():
            logger.info(f"Saving result to {midi_save_path}")
            midi = convert_msgs_to_midi(msgs=msgs)
            midi.save(midi_save_path)

        curr_midi_channel += 1
        if curr_midi_channel == 9:  # Skip drum channel
            curr_midi_channel += 1

        control_sentinel.clear()
        if reset_sentinel.is_set():
            return
        # A reset during generation: loop back so the top clears + re-listens.
        if soft_reset_sentinel.is_set():
            continue
        currently_generating_sentinel.clear()
        msgs, prev_context, _, num_active_pitches = capture_and_update_kv(
            model=model,
            msgs=msgs,
            prev_context=prev_context,
            control_sentinel=control_sentinel,
            reset_sentinel=reset_sentinel,
            wait_for_close=wait_for_close,
            midi_performance_queue=midi_performance_queue,
            midi_capture_channel=curr_midi_channel,
            first_msg_epoch_time_ms=first_on_msg_epoch_ms,
        )

    keypress_thread.join()
    midi_control_thread.join()
    if play_file_thread:
        play_file_thread.join()


def _next_duet_channel(channel: int) -> int:
    """Cycle 1..15, skipping 9 (drums) and wrapping. Channel 0 is reserved for
    the very first capture window."""
    channel += 1
    if channel == 9:  # Skip drum channel
        channel += 1
    if channel > 15:
        channel = 1
    return channel


def run_duet(
    model: TransformerLM,
    midi_performance_queue: queue.Queue,
    midi_control_queue: queue.Queue,
    midi_out_port: str | None,
    midi_save_path: str | None,
    midi_reset_control_signal: int,
    reset_sentinel: threading.Event,
    temperature: float,
    min_p: float,
    listen_ms: int,
    play_ms: int,
):
    """Experimental duet mode: you and the model perform *together*.

    Rather than waiting for an explicit hand-off, this alternates very short
    capture and generation windows in a tight loop. Two things make it a duet
    rather than fast turn-taking:

      1. The capture phase does NOT flush pending input (flush_pending=False),
         so notes you play *during* the model's burst are preserved.
      2. Both your captured notes and the model's generated notes accumulate in
         `msgs`, which is re-tokenized as the priming context for every burst -
         so each side hears the other with roughly one window of latency.

    Reset (CC reset signal / a letter + Enter) clears the shared context and
    restarts with a clean channel cycle.
    """
    logger = get_logger("DUET")
    tokenizer = AbsTokenizer(config_path=TOKENIZER_CONFIG_PATH)
    control_sentinel = threading.Event()
    currently_generating_sentinel = threading.Event()

    if midi_out_port:
        close_notes(midi_out_port)

    # Only the reset signal is meaningful in duet mode (takeover is automatic),
    # so disable the takeover CC by passing midi_control_signal=None.
    midi_control_thread = threading.Thread(
        target=listen_for_midi_control_signal,
        kwargs={
            "midi_control_queue": midi_control_queue,
            "control_sentinel": control_sentinel,
            "reset_sentinel": reset_sentinel,
            "currently_generating_sentinel": currently_generating_sentinel,
            "midi_control_signal": None,
            "midi_reset_control_signal": midi_reset_control_signal,
            "back_and_forth": True,
        },
        daemon=True,
    )
    keypress_thread = threading.Thread(
        target=listen_for_keypress_control_signal,
        kwargs={
            "control_sentinel": control_sentinel,
            "reset_sentinel": reset_sentinel,
            "currently_generating_sentinel": currently_generating_sentinel,
            "back_and_forth": True,
        },
        daemon=True,
    )
    midi_control_thread.start()
    keypress_thread.start()

    def _window(window_ms: int) -> threading.Timer:
        # Bound the next phase to window_ms by setting the control sentinel.
        control_sentinel.clear()
        timer = threading.Timer(window_ms / 1000.0, control_sentinel.set)
        timer.daemon = True
        timer.start()
        return timer

    logger.info(
        f"Duet mode started (listen={listen_ms}ms, play={play_ms}ms). "
        "Play along; reset to start fresh."
    )

    # Initial capture: flush once so we begin from a clean slate.
    timer = _window(listen_ms)
    msgs, prev_context, first_on_msg_epoch_ms, num_active_pitches = (
        capture_and_update_kv(
            model=model,
            msgs=[],
            prev_context=[],
            control_sentinel=control_sentinel,
            reset_sentinel=reset_sentinel,
            wait_for_close=False,
            midi_performance_queue=midi_performance_queue,
            midi_capture_channel=0,
            flush_pending=True,
        )
    )
    timer.cancel()

    curr_midi_channel = 1
    while not reset_sentinel.is_set():
        # The model needs something to continue from. Until you've played a
        # note, keep listening (channel 0) rather than generating from nothing
        # (which would also crash convert_msgs_to_midi on an empty timeline).
        if not msgs or first_on_msg_epoch_ms is None:
            timer = _window(listen_ms)
            msgs, prev_context, first_on_msg_epoch_ms, num_active_pitches = (
                capture_and_update_kv(
                    model=model,
                    msgs=msgs,
                    prev_context=prev_context,
                    control_sentinel=control_sentinel,
                    reset_sentinel=reset_sentinel,
                    wait_for_close=False,
                    midi_performance_queue=midi_performance_queue,
                    midi_capture_channel=0,
                    flush_pending=False,
                )
            )
            timer.cancel()
            continue

        # --- Model burst ---
        # The burst self-terminates after play_ms of token generation via
        # max_gen_ms (a timer can't bound it: decode_tokens clears the control
        # sentinel on entry). control_sentinel is left for reset handling only.
        currently_generating_sentinel.set()
        control_sentinel.clear()
        timing = {}
        msgs = stream_msgs(
            model=model,
            tokenizer=tokenizer,
            msgs=msgs,
            prev_context=prev_context,
            midi_output_port=midi_out_port,
            first_on_msg_epoch_ms=first_on_msg_epoch_ms,
            control_sentinel=control_sentinel,
            temperature=temperature,
            min_p=min_p,
            num_preceding_active_pitches=num_active_pitches,
            midi_stream_channel=curr_midi_channel,
            is_ending=False,
            max_gen_ms=play_ms,
            timing_out=timing,
        )
        currently_generating_sentinel.clear()

        if midi_save_path:
            # Best-effort: reused channels in long duets can yield out-of-order
            # per-track deltas that mido's file writer rejects (negative time).
            # The in-memory timeline still works; just skip the snapshot on error.
            try:
                convert_msgs_to_midi(msgs=msgs).save(midi_save_path)
            except ValueError as e:
                logger.warning(f"Skipping duet MIDI snapshot: {e}")

        curr_midi_channel = _next_duet_channel(curr_midi_channel)
        if reset_sentinel.is_set():
            break

        # --- Player capture window (no flush: keep what you played) ---
        # The model schedules notes slightly ahead of wall-clock. Hold the
        # capture window open until those notes have played out (plus a small
        # buffer), so: (a) you play *over* the model's phrase and it's captured,
        # and (b) the context's last onset is in the past before the next burst
        # (which keeps decode_tokens_to_midi's precondition satisfied). Bounded
        # below by listen_ms and above by DUET_MAX_CATCHUP_MS.
        window_ms = listen_ms
        last_epoch_ms = timing.get("last_epoch_ms")
        if last_epoch_ms is not None:
            catchup_ms = last_epoch_ms + DUET_CATCHUP_BUFFER_MS - get_epoch_time_ms()
            window_ms = max(listen_ms, min(int(catchup_ms), DUET_MAX_CATCHUP_MS))

        timer = _window(window_ms)
        msgs, prev_context, _, num_active_pitches = capture_and_update_kv(
            model=model,
            msgs=msgs,
            prev_context=prev_context,
            control_sentinel=control_sentinel,
            reset_sentinel=reset_sentinel,
            wait_for_close=False,
            midi_performance_queue=midi_performance_queue,
            midi_capture_channel=curr_midi_channel,
            first_msg_epoch_time_ms=first_on_msg_epoch_ms,
            flush_pending=False,
        )
        timer.cancel()
        curr_midi_channel = _next_duet_channel(curr_midi_channel)

    midi_control_thread.join()
    keypress_thread.join()


def insert_embedding(
    model: TransformerLM,
    embedding_model_checkpoint_path: str,
    embedding_midi_path: str,
):
    logger = get_logger()
    logger.info(f"Loading embedding from {embedding_midi_path}")
    emb = _get_embedding(
        embedding_model_checkpoint_path=embedding_model_checkpoint_path,
        embedding_midi_path=embedding_midi_path,
    )
    logger.info(f"Inserting embedding into context")
    model.fill_condition_kv(mx.array([emb], dtype=DTYPE))

    global EMBEDDING_OFFSET
    EMBEDDING_OFFSET = 1


def forward_midi_input_port(
    midi_input_port: str,
    midi_control_queue: queue.Queue,
    midi_performance_queue: queue.Queue | None,
):
    logger = get_logger("MIDI-FORWARD")
    logger.info(f"Forwarding MIDI from port: '{midi_input_port}'")

    if midi_performance_queue is None:
        logger.info(
            f"MIDI file provided - only forwarding {midi_input_port} to control queue"
        )

    try:
        with mido.open_input(midi_input_port) as midi_in:
            while True:
                msg = midi_in.receive(block=True)
                if msg:
                    midi_control_queue.put(msg)
                    if midi_performance_queue is not None:
                        midi_performance_queue.put(msg)

    except (Exception, KeyboardInterrupt) as e:
        logger.error(f"Error in MIDI forwarder: {e}")
    finally:
        logger.info("MIDI forwarder has shut down.")


def main(args):
    logger = get_logger()
    # Apply CLI-configurable globals before the model cache is built / used.
    global TURN_SWITCH_MODE, TURN_FREEZE_LATENCY_MS, MAX_SEQ_LEN
    TURN_SWITCH_MODE = args.turn_switch_mode
    TURN_FREEZE_LATENCY_MS = args.turn_freeze_latency_ms
    MAX_SEQ_LEN = args.max_seq_len
    logger.info(f"Max sequence length (context window): {MAX_SEQ_LEN} tokens")
    logger.info(
        f"Turn-switch mode: {TURN_SWITCH_MODE}"
        + (
            f" (freeze latency {TURN_FREEZE_LATENCY_MS}ms)"
            if TURN_SWITCH_MODE == "freeze"
            else ""
        )
    )
    # Create the output port up front so a virtual port (e.g. the Pianoteq
    # target) is visible to other apps immediately and for the whole session.
    if args.midi_out and args.midi_out not in mido.get_output_names():
        open_output(args.midi_out)
        logger.info(f"Created persistent virtual MIDI port: '{args.midi_out}'")
    global EMBEDDING_OFFSET
    model = load_model(checkpoint_path=args.checkpoint)
    # Don't exceed the model's positional capacity (RoPE is built for this).
    model_max = getattr(model.model_config, "max_seq_len", MAX_SEQ_LEN)
    if MAX_SEQ_LEN > model_max:
        logger.warning(
            f"--max_seq_len {MAX_SEQ_LEN} exceeds the model's trained max "
            f"{model_max}; clamping to {model_max}."
        )
        MAX_SEQ_LEN = model_max
    model = warmup_model(model=model)
    if args.embedding_checkpoint and args.embedding_midi_path:
        insert_embedding(
            model=model,
            embedding_model_checkpoint_path=args.embedding_checkpoint,
            embedding_midi_path=args.embedding_midi_path,
        )
    else:
        model.fill_condition_kv(
            mx.zeros((1, model.model_config.emb_size), dtype=DTYPE)
        )
        EMBEDDING_OFFSET = 1

    assert (args.midi_path and os.path.isfile(args.midi_path)) or args.midi_in

    logger.info(f"Available MIDI ports: {mido.get_output_names()}")
    midi_performance_queue = queue.Queue()
    midi_control_queue = queue.Queue()

    if args.midi_in:
        forwarder_thread = threading.Thread(
            target=forward_midi_input_port,
            kwargs={
                "midi_input_port": args.midi_in,
                "midi_control_queue": midi_control_queue,
                "midi_performance_queue": (
                    midi_performance_queue if args.midi_path is None else None
                ),
            },
            daemon=True,
        )
        forwarder_thread.start()

    if args.control_midi_in:
        # Dedicated control-signal port (e.g. MC8 footswitch). Forward only to
        # the control queue with no performance queue, so its CCs trigger
        # takeover/reset but are never tokenized as notes.
        control_forwarder_thread = threading.Thread(
            target=forward_midi_input_port,
            kwargs={
                "midi_input_port": args.control_midi_in,
                "midi_control_queue": midi_control_queue,
                "midi_performance_queue": None,
            },
            daemon=True,
        )
        control_forwarder_thread.start()

    reset_sentinel = threading.Event()
    while True:
        if args.duet:
            run_duet(
                model=model,
                midi_performance_queue=midi_performance_queue,
                midi_control_queue=midi_control_queue,
                midi_out_port=args.midi_out,
                midi_save_path=args.save_path,
                midi_reset_control_signal=args.midi_reset_control_signal,
                reset_sentinel=reset_sentinel,
                temperature=args.temp,
                min_p=args.min_p,
                listen_ms=args.duet_listen_ms,
                play_ms=args.duet_play_ms,
            )
        else:
            run(
                model=model,
                midi_performance_queue=midi_performance_queue,
                midi_control_queue=midi_control_queue,
                midi_through_port=args.midi_through,
                midi_out_port=args.midi_out,
                midi_path=args.midi_path,
                midi_save_path=args.save_path,
                midi_control_signal=args.midi_control_signal,
                midi_reset_control_signal=args.midi_reset_control_signal,
                reset_sentinel=reset_sentinel,
                wait_for_close=args.wait_for_close,
                temperature=args.temp,
                min_p=args.min_p,
                back_and_forth=args.back_and_forth,
            )
        reset_sentinel = threading.Event()


def playback(midi_path: str, midi_out: str, save_path: str | None = None):
    # Mocks generated playback by streaming from a real MIDI file

    close_notes(midi_out)
    starting_epoch_time_ms = get_epoch_time_ms()
    tokenizer = AbsTokenizer(config_path=TOKENIZER_CONFIG_PATH)
    tokens_queue = queue.Queue()
    midi_messages_queue = queue.Queue()
    stream_midi_results_queue = queue.Queue()
    control_sentinel = threading.Event()

    midi_dict = MidiDict.from_midi(midi_path)
    midi_dict.remove_redundant_pedals()
    tokenized_sequence = tokenizer.tokenize(
        midi_dict,
        add_dim_tok=False,
        remove_preceding_silence=False,
    )
    tokenized_sequence = tokenized_sequence[
        tokenized_sequence.index(tokenizer.bos_tok) + 1 :
    ]

    # Populate token queue synthetically
    for tok in tokenized_sequence:
        tokens_queue.put(tok)

    decode_tokens_to_midi_thread = threading.Thread(
        target=decode_tokens_to_midi,
        kwargs={
            "generated_tokens_queue": tokens_queue,
            "outbound_midi_msg_queue": midi_messages_queue,
            "tokenizer": tokenizer,
            "first_on_msg_epoch_ms": starting_epoch_time_ms,
            "priming_seq_last_onset_ms": 0,
        },
    )
    decode_tokens_to_midi_thread.start()

    stream_midi_thread = threading.Thread(
        target=stream_midi,
        kwargs={
            "inbound_midi_msg_queue": midi_messages_queue,
            "msgs": [],
            "last_channel_msg_epoch_time_ms": starting_epoch_time_ms,
            "midi_output_port": midi_out,
            "control_sentinel": control_sentinel,
            "midi_stream_channel": 0,
            "results_queue": stream_midi_results_queue,
        },
    )
    stream_midi_thread.start()

    decode_tokens_to_midi_thread.join()
    stream_midi_thread.join()
    msgs = stream_midi_results_queue.get()
    mid = convert_msgs_to_midi(msgs)

    if save_path is not None:
        mid.save(save_path)

    return msgs


def close_notes(midi_out_port: str):
    with open_output(midi_out_port) as out:
        out.send(mido.Message(type="control_change", control=64, value=0))
        for note in range(128):
            out.send(mido.Message("note_off", note=note, velocity=0))


if __name__ == "__main__":
    args = parse_args()

    if args.hardware:
        set_calibration_settings(args.hardware)

    if args.playback is True:
        # Playback only mode for testing
        assert args.midi_path is not None, "Must provide midi_path"
        try:
            playback(
                midi_path=args.midi_path,
                midi_out=args.midi_out,
                save_path=args.save_path,
            )
        except KeyboardInterrupt:
            close_notes(args.midi_out)
    else:
        try:
            main(args)
        except KeyboardInterrupt:
            close_notes(args.midi_out)
