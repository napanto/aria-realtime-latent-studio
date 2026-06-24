"""Importable wrapper around the vendored real-time MLX demo.

The two plain-Aria backends (``aria_base``, ``aria_jazz``) are *identical* at
run time except for (i) the checkpoint and (ii) the tokenizer config. Both run
through :mod:`realtime.aria_demo_mlx`, which is the proven EleutherAI/aria
real-time engine copied verbatim (KV-cache, chunked prefill, duration
recalculation, beam-of-3 first-onset, min-p sampling, low-latency MIDI
scheduling). We do **not** reimplement any of that here.

What this module adds is a small object the GUI / CLI can drive:

  >>> eng = RealtimeAriaEngine(RealtimeConfig(
  ...     checkpoint="weights/aria_jazz/model.safetensors",
  ...     tokenizer_config="weights/aria_jazz/tokenizer-config.json",
  ...     midi_in="My Keyboard", midi_out="My Synth"))
  >>> eng.load()        # blocking: load + warm-up MLX graphs
  >>> eng.start()       # spawn the demo run-loop in a background thread
  >>> eng.stop()        # signal the run-loop to exit

NOTE: this requires ``mlx`` + the ``aria`` package + ``ariautils`` to be
importable, which is only true on Apple Silicon (see README). On Linux the
import of :mod:`realtime.aria_demo_mlx` will fail — that is expected and is why
this is a *thin* wrapper kept separate from the model registry.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RealtimeConfig:
    checkpoint: str
    tokenizer_config: str
    midi_out: str
    midi_in: Optional[str] = None
    midi_through: Optional[str] = None
    midi_path: Optional[str] = None         # play a file instead of a live port
    save_path: Optional[str] = None
    temperature: float = 0.95               # demo default
    min_p: float = 0.03
    quantize: bool = False
    wait_for_close: bool = False
    hardware: Optional[str] = None          # path to calibration json
    # AI-takeover / reset CC numbers (optional)
    midi_control_signal: Optional[int] = None
    midi_reset_control_signal: Optional[int] = None
    back_and_forth: bool = False


class RealtimeAriaEngine:
    """Owns one warmed-up MLX model and a background run-loop thread."""

    def __init__(self, cfg: RealtimeConfig):
        self.cfg = cfg
        self._model = None
        self._tokenizer = None
        self._demo = None  # the imported aria_demo_mlx module
        self._thread: Optional[threading.Thread] = None
        self._reset_sentinel: Optional[threading.Event] = None
        self._midi_performance_queue: "queue.Queue" = queue.Queue()
        self._midi_control_queue: "queue.Queue" = queue.Queue()
        self._loaded = False

    # -- lifecycle ---------------------------------------------------------
    def _import_demo(self):
        if self._demo is None:
            # Imported lazily so the registry / GUI import cleanly on Linux.
            from realtime import aria_demo_mlx as demo

            self._demo = demo
        return self._demo

    def load(self):
        """Load the checkpoint and warm up the MLX compute graphs (blocking)."""
        demo = self._import_demo()

        if not Path(self.cfg.checkpoint).exists():
            raise FileNotFoundError(
                f"checkpoint not found: {self.cfg.checkpoint}. "
                "Run scripts/download_models.py first."
            )

        # The vendored demo reads its tokenizer config from a module-level path
        # constant. Point it at *this* model's config so aria_base and
        # aria_jazz can each use their own tokenizer.
        demo.TOKENIZER_CONFIG_PATH = Path(self.cfg.tokenizer_config)
        if self.cfg.hardware:
            demo.set_calibration_settings(self.cfg.hardware)

        # Mirror demo_mlx.main()'s arg object for quantize.
        demo.args = _DemoArgs(self.cfg)

        self._model = demo.load_model(checkpoint_path=self.cfg.checkpoint)
        self._model = demo.warmup_model(model=self._model)

        # Match main(): with no conditioning embedding we fill a zero condition
        # KV slot so EMBEDDING_OFFSET == 1 (the demo's unconditional path).
        import mlx.core as mx

        self._model.fill_condition_kv(
            mx.zeros((1, self._model.model_config.emb_size), dtype=demo.DTYPE)
        )
        demo.EMBEDDING_OFFSET = 1

        from ariautils.tokenizer import AbsTokenizer

        self._tokenizer = AbsTokenizer(config_path=Path(self.cfg.tokenizer_config))
        self._loaded = True
        return self

    def start(self):
        """Spawn the demo run-loop in a background thread."""
        if not self._loaded:
            raise RuntimeError("call load() before start()")
        if self._thread is not None and self._thread.is_alive():
            return self
        demo = self._import_demo()
        self._reset_sentinel = threading.Event()

        # If a live input port is configured, start the demo's forwarder.
        if self.cfg.midi_in:
            fwd = threading.Thread(
                target=demo.forward_midi_input_port,
                kwargs={
                    "midi_input_port": self.cfg.midi_in,
                    "midi_control_queue": self._midi_control_queue,
                    "midi_performance_queue": (
                        self._midi_performance_queue
                        if self.cfg.midi_path is None
                        else None
                    ),
                },
                daemon=True,
            )
            fwd.start()

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        return self

    def _run_loop(self):
        demo = self._import_demo()
        cfg = self.cfg
        demo.run(
            model=self._model,
            midi_performance_queue=self._midi_performance_queue,
            midi_control_queue=self._midi_control_queue,
            midi_through_port=cfg.midi_through,
            midi_out_port=cfg.midi_out,
            midi_path=cfg.midi_path,
            midi_save_path=cfg.save_path,
            midi_control_signal=cfg.midi_control_signal,
            midi_reset_control_signal=cfg.midi_reset_control_signal,
            reset_sentinel=self._reset_sentinel,
            wait_for_close=cfg.wait_for_close,
            temperature=cfg.temperature,
            min_p=cfg.min_p,
            back_and_forth=cfg.back_and_forth,
        )

    def trigger_ai_takeover(self):
        """Push a synthetic control signal so the AI starts generating now."""
        # The demo's keypress listener treats an empty stdin line as 'go'; from
        # code we set the control sentinel via the reset machinery. The simplest
        # robust trigger is the configured MIDI control CC if present.
        if self._reset_sentinel is None:
            return
        # No-op placeholder: real triggering happens via the MIDI control CC or
        # the keyboard listener inside demo.run. See STATUS.md (GUI transport).

    def stop(self):
        """Signal the run-loop to exit and join it."""
        if self._reset_sentinel is not None:
            self._reset_sentinel.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


class _DemoArgs:
    """Minimal stand-in for argparse.Namespace that demo.load_model reads."""

    def __init__(self, cfg: RealtimeConfig):
        self.quantize = cfg.quantize
