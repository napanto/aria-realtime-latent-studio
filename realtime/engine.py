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
    # An already-open mido output port (e.g. a server-held *virtual* output the
    # file-generate path also sends through). When set, the run loop streams
    # through THIS object under the ``midi_out`` name instead of opening/creating
    # its own port — so realtime and file playback share one persistent CoreMIDI
    # source and downstream apps (Pianoteq) stay connected across turns.
    external_out_port: Optional[object] = None
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
    # Turn-switch timing ("snap" | "freeze" | "catchup"). None keeps the demo
    # default (catchup), which is best for tight live keyboard turn-taking. For
    # file-seeded / sparse use, "snap" (insert the elapsed pause as a rest, then
    # play the continuation forward) streams more reliably.
    turn_switch_mode: Optional[str] = None


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
        """Load the checkpoint and warm up the MLX compute graphs (blocking).

        Loads the demo checkpoint, warms up the compute graphs, and fills a zero
        condition KV slot (EMBEDDING_OFFSET = 1, the demo's unconditional path).
        """
        demo = self._import_demo()

        # The vendored demo reads its tokenizer config from a module-level path
        # constant. Point it at *this* model's config so aria_base and
        # aria_jazz can each use their own.
        demo.TOKENIZER_CONFIG_PATH = Path(self.cfg.tokenizer_config)
        if self.cfg.hardware:
            demo.set_calibration_settings(self.cfg.hardware)
        if self.cfg.turn_switch_mode:
            demo.TURN_SWITCH_MODE = self.cfg.turn_switch_mode

        # Mirror demo_mlx.main()'s arg object for quantize.
        demo.args = _DemoArgs(self.cfg)

        import mlx.core as mx
        from ariautils.tokenizer import AbsTokenizer

        if not Path(self.cfg.checkpoint).exists():
            raise FileNotFoundError(
                f"checkpoint not found: {self.cfg.checkpoint}. "
                "Run scripts/download_models.py first."
            )

        self._model = demo.load_model(checkpoint_path=self.cfg.checkpoint)
        self._model = demo.warmup_model(model=self._model)

        # Match main(): with no conditioning embedding we fill a zero condition
        # KV slot so EMBEDDING_OFFSET == 1 (the demo's unconditional path).
        self._model.fill_condition_kv(
            mx.zeros((1, self._model.model_config.emb_size), dtype=demo.DTYPE)
        )
        demo.EMBEDDING_OFFSET = 1

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

        # Route the run loop through a server-held virtual output if one was
        # provided: register it under midi_out so demo.open_output(midi_out)
        # returns a _KeepAlive view of THIS port (sends, never closes/recreates).
        # This is what makes "Takeover" stream through the same 'Aria Studio'
        # source the file-generate path uses, instead of a second (stale-ID) one.
        if self.cfg.external_out_port is not None:
            demo._persistent_virtual_ports[self.cfg.midi_out] = (
                self.cfg.external_out_port
            )

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

    # -- live stream status ------------------------------------------------
    def current_lag_ms(self) -> float:
        """How far behind real-time the live stream currently is (ms). >0 means
        the decode is slower than real-time and the playback timeline has been
        stretched to keep notes flowing (rather than dropping them)."""
        demo = self._import_demo()
        return float(getattr(demo, "CURRENT_STREAM_LAG_MS", 0.0))

    def trigger_ai_takeover(self):
        """Inject a synthetic takeover control signal so generation starts now.

        Mirrors a foot-controller hand-over: a ``note_on`` (which primes the
        demo's control listener's ``seen_note_on`` gate) followed by the takeover
        CC (value 127). Both go to the CONTROL queue only — they are never
        tokenized as performance input. demo.run()'s ``listen_for_midi_control_
        signal`` then sets its control sentinel, ending the capture turn and
        starting generation. Requires the engine to have been started with a
        concrete ``midi_control_signal`` (the start endpoint defaults it to 102).
        """
        if not self.is_running:
            raise RuntimeError("engine is not running; call start() first")
        cc = self.cfg.midi_control_signal
        if cc is None:
            raise RuntimeError(
                "no midi_control_signal configured; cannot trigger takeover"
            )
        import mido

        self._midi_control_queue.put(
            mido.Message("note_on", note=60, velocity=64)
        )
        self._midi_control_queue.put(
            mido.Message("control_change", control=int(cc), value=127)
        )

    def trigger_reset(self):
        """Inject the reset control CC so the run loop performs a SOFT reset:
        clear the captured context (and any in-flight generation) and resume
        listening for notes — the engine keeps running. Mirrors the footswitch
        reset CC; used by the GUI reset button. Hard stop is stop()."""
        if not self.is_running:
            raise RuntimeError("engine is not running; call start() first")
        cc = self.cfg.midi_reset_control_signal
        if cc is None:
            raise RuntimeError(
                "no midi_reset_control_signal configured; cannot reset"
            )
        import mido

        self._midi_control_queue.put(
            mido.Message("control_change", control=int(cc), value=127)
        )

    def stop(self):
        """Signal the run-loop to exit and join it.

        Injects the reset control CC first: ``decode_tokens`` only watches the
        demo's *control* sentinel (set by the control listener on the reset CC),
        not ``reset_sentinel``. Without this an in-flight generation keeps
        streaming and driving MLX after stop(), which then collides with any
        later MLX op (e.g. switching to another model) and crashes the process. With
        it, generation halts within ~one token so MLX is quiescent on return.
        """
        cc = self.cfg.midi_reset_control_signal
        if cc is not None and self.is_running:
            import mido

            for _ in range(3):
                self._midi_control_queue.put(
                    mido.Message("control_change", control=int(cc), value=127)
                )
        if self._reset_sentinel is not None:
            self._reset_sentinel.set()
        if self._thread is not None:
            self._thread.join(timeout=8.0)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


class _DemoArgs:
    """Minimal stand-in for argparse.Namespace that demo.load_model reads."""

    def __init__(self, cfg: RealtimeConfig):
        self.quantize = cfg.quantize
