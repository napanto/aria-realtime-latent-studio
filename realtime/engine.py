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
    # Turn-switch timing ("snap" | "freeze" | "catchup"). None keeps the demo
    # default (catchup), which is best for tight live keyboard turn-taking. For
    # file-seeded / sparse use, "snap" (insert the elapsed pause as a rest, then
    # play the continuation forward) streams more reliably.
    turn_switch_mode: Optional[str] = None
    # --- AriaVAE latent mode (set latent_dir to turn it on) ---------------
    # When latent_dir is set, the VAE *decoder* becomes the streamed model:
    # an 8-token z-prefix occupies KV positions 0..K-1 plus a per-layer
    # z-residual is added before every block. The demo's prefill/decode_one
    # branch on the module-level ``LATENT`` so the SAME real-time loop streams
    # AriaVAE. latent_seed_midi anchors the base z to a style; latent_prior
    # samples z~N(0,I) instead.
    latent_dir: Optional[str] = None
    latent_seed_midi: Optional[str] = None
    latent_prior: bool = False
    latent_gain: float = 2.0


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

        Two paths:
          * plain Aria (``latent_dir`` unset): load the demo checkpoint, warm up,
            fill a zero condition KV slot (EMBEDDING_OFFSET = 1).
          * AriaVAE (``latent_dir`` set): build the VAE via the demo's
            ``load_latent`` + ``set_latent_base``; the VAE *decoder* becomes the
            streamed model and ``demo.LATENT`` makes prefill/decode_one route
            through the per-layer z-residual + 8-token z-prefix. Mirrors exactly
            demo_mlx.main()'s ``args.latent_dir`` branch.
        """
        demo = self._import_demo()

        # The vendored demo reads its tokenizer config from a module-level path
        # constant. Point it at *this* model's config so aria_base and
        # aria_jazz (and the AriaVAE demo tokenizer) can each use their own.
        demo.TOKENIZER_CONFIG_PATH = Path(self.cfg.tokenizer_config)
        if self.cfg.hardware:
            demo.set_calibration_settings(self.cfg.hardware)
        if self.cfg.turn_switch_mode:
            demo.TURN_SWITCH_MODE = self.cfg.turn_switch_mode

        # Mirror demo_mlx.main()'s arg object for quantize.
        demo.args = _DemoArgs(self.cfg)

        import mlx.core as mx
        from ariautils.tokenizer import AbsTokenizer

        if self.cfg.latent_dir:
            self._load_latent(demo)
            self._tokenizer = AbsTokenizer(
                config_path=Path(self.cfg.tokenizer_config)
            )
            self._loaded = True
            return self

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

    def _load_latent(self, demo):
        """Build the AriaVAE latent and arm the demo's latent path.

        Byte-for-byte mirror of demo_mlx.main()'s ``args.latent_dir`` branch:
        load_latent -> setup_stream -> set_latent_base (sets EMBEDDING_OFFSET=K
        and the base z) -> warmup_model -> prefill_prefix (write the z-prefix
        into KV positions 0..K-1). After this, demo.run() streams AriaVAE because
        demo.prefill / demo.decode_one branch on demo.LATENT.
        """
        from ariautils.tokenizer import AbsTokenizer

        tokenizer = AbsTokenizer(config_path=Path(self.cfg.tokenizer_config))
        demo.LATENT, demo.LATENT_CTRL = demo.load_latent(
            self.cfg.latent_dir, tokenizer, self.cfg.quantize, self.cfg.latent_gain
        )
        # The GUI drives the latent live over HTTP (/api/realtime/latent), not
        # via hardware CC sliders, so no SLIDER_MAP is mapped here.
        demo.SLIDER_MAP = {}

        model = demo.LATENT.decoder
        model_max = getattr(model.model_config, "max_seq_len", demo.MAX_SEQ_LEN)
        if demo.MAX_SEQ_LEN > model_max:
            demo.MAX_SEQ_LEN = model_max
        demo.LATENT.setup_stream(demo.MAX_SEQ_LEN)
        demo.set_latent_base(
            demo.LATENT, demo.LATENT_CTRL, tokenizer,
            self.cfg.latent_seed_midi, self.cfg.latent_prior,
        )  # sets demo.EMBEDDING_OFFSET = K
        model = demo.warmup_model(model=model)
        demo.LATENT.prefill_prefix()  # write z-prefix into KV positions 0..K-1
        self._model = model

        # The latent forward (_latent_forward) bypasses Transformer.__call__, so
        # model.kv_ctx is never initialised and stays None. The demo's debug
        # logging eagerly evaluates ``tokenizer.decode(model.get_kv_ctx())`` in
        # recalc_dur_tokens_chunked / decode_first_tokens, which raises on None
        # and silently kills the generate thread (no streaming). Initialise
        # kv_ctx exactly as Transformer.__call__ would (all-unk), so get_kv_ctx()
        # returns [] and the debug path is a harmless no-op. Debug-only state;
        # the real KV cache lives in the per-layer KVCache objects.
        import mlx.core as mx

        model.model.kv_ctx = mx.full(model.max_seq_len, 3)

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

    # -- latent (AriaVAE) live control ------------------------------------
    @property
    def is_latent(self) -> bool:
        return bool(self.cfg.latent_dir)

    def set_latent_offsets(self, offsets: dict) -> list:
        """Move the live latent z by per-attribute deltas (AriaVAE only).

        ``offsets`` maps attribute name -> Δ (raw attribute units). For each we
        call ``LATENT_CTRL.set_attr_delta(idx, Δ)`` then STAGE the new z via
        ``LATENT.request_z(LATENT_CTRL.z())`` (numpy only — no MLX on this HTTP
        thread). The decoder thread materialises prefix+residuals on its next
        token, so the player hears the change within ~one token and MLX is never
        driven from two threads at once (which segfaults). Returns the list of
        applied attribute names.
        """
        demo = self._import_demo()
        if demo.LATENT is None or demo.LATENT_CTRL is None:
            raise RuntimeError("this engine is not running in AriaVAE latent mode")

        ctrl = demo.LATENT_CTRL
        applied = []
        for name, delta in (offsets or {}).items():
            if name not in ctrl.names:
                continue
            ctrl.set_attr_delta(ctrl.attr_index(name), float(delta))
            applied.append(name)
        demo.LATENT.request_z(ctrl.z())  # numpy stage; decoder thread applies it
        return applied

    def latent_attributes(self) -> list:
        """Slider spec for the running AriaVAE: [{name, label, r2, active}]."""
        demo = self._import_demo()
        if demo.LATENT_CTRL is None:
            return []
        from latent.attributes import ATTR_LABELS

        ctrl = demo.LATENT_CTRL
        active = set(ctrl.active)
        out = []
        for k, name in enumerate(ctrl.names):
            r2 = float(ctrl.r2[k])
            out.append({
                "name": name,
                "label": ATTR_LABELS.get(name, name),
                "r2": round(r2, 3) if r2 == r2 else None,  # NaN -> None
                "active": k in active,
            })
        return out

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

    def stop(self):
        """Signal the run-loop to exit and join it.

        Injects the reset control CC first: ``decode_tokens`` only watches the
        demo's *control* sentinel (set by the control listener on the reset CC),
        not ``reset_sentinel``. Without this an in-flight generation keeps
        streaming and driving MLX after stop(), which then collides with any
        later MLX op (e.g. switching to Cadenza) and crashes the process. With
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
