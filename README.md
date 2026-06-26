# Aria Realtime Studio

A macOS / Apple-Silicon **real-time piano-continuation** app: play a few bars on
a MIDI keyboard, hand the turn to the model, and it continues your phrase
note-by-note, low-latency, straight to a Core MIDI output. Two Aria
piano-performance models run behind one browser UI, both on **MLX** (Apple
Silicon), both with **grammar-constrained decoding**.

> This project began life as a VAE latent-control studio (hence the repository
> name `aria-realtime-studio`); the latent models have since been retired,
> leaving this focused real-time Aria continuation app.

| key | model | backend | role |
|-----|-------|---------|------|
| `aria_base` | Aria (original)        | MLX (real-time engine) | upstream EleutherAI/aria real-time demo weights |
| `aria_jazz` | Aria (jazz fine-tuned) | MLX (real-time engine) | jazz fine-tune, already MLX-converted on the Hub |

Both models are autoregressive transformers driven by the *proven* EleutherAI/aria
real-time MLX demo (KV-cache, chunked prefill, duration recalculation, beam-of-3
first onset, min-p sampling, low-latency MIDI scheduling) — vendored verbatim and
wrapped in an importable engine. They differ only in checkpoint and tokenizer
config.

**Grammar-constrained decoding.** The Aria `AbsTokenizer` asserts a strict
note/pedal structure (every note is `instrument,pitch,vel` → `onset` → `dur`;
every pedal → `onset`). A note/pedal FSM (`studio/mlx_vae/grammar.py`) masks each
step's logits to the grammatically valid next category, so the model can never
emit a malformed group that would crash detokenize mid-stream. It degrades to
structural-only masking if the grammar module can't be imported, so it never
breaks generation.

---

## Architecture

```
                   ┌────────────────── gui/ (FastAPI single-page app) ─────────────┐
                   │   model selector · transport (start/takeover/reset) · ports   │
                   └───────────────────────────────┬───────────────────────────────┘
                                                   │
                                                   ▼
   realtime/  RealtimeAriaEngine ── wraps ──► realtime/aria_demo_mlx.py (vendored)
     load() · start() · stop()                MIDI-in → continuation → MIDI-out,
                                              low-latency, KV-cache, grammar FSM
                                                   │
   studio/mlx_vae/grammar.py ── note/pedal FSM the real-time decoder masks with ───┘

   models/registry.py  ── single source of truth for checkpoints / tokenizers / backends ──┐
   scripts/download_models.py ── reads the same registry, fetches weights from HF ──────────┘
```

- **`models/`** — model-abstraction layer. `registry.py` declares, per model, the
  HF coordinates, the local `weights/` path, the backend, and the tokenizer
  config. Both the runtime and the downloader read this one table. It imports no
  torch/mlx, so it is cheap to import anywhere.
- **`realtime/`** — the real-time MLX engine. `aria_demo_mlx.py` is the upstream
  demo copied verbatim; `engine.py` is a thin `RealtimeAriaEngine` that points it
  at a chosen checkpoint + tokenizer and runs its loop in a background thread.
  `calibrate.py` is an optional hardware-latency calibration tool.
- **`studio/mlx_vae/`** — `grammar.py`, the note/pedal grammar FSM the real-time
  decoder uses for grammar-constrained decoding.
- **`gui/`** — the local web app (FastAPI backend + one self-contained
  `index.html`).
- **`scripts/`** — `download_models.py` (fetch weights from the Hub).

---

## GUI stack choice — and why

**A local web app: FastAPI (Python) backend + a single vanilla-JS page.**
Considered alternatives were native (PyQt / rumps). Rationale:

1. **Compute stays in Python.** The models run on MLX (Apple Silicon), which is
   Python. A web UI keeps the heavy generation in-process and only ships control
   over localhost — no IPC bridge to a native toolkit.
2. **Zero-build controls.** HTML range inputs and `<select>`s need no toolkit, no
   `.app` bundling, no codesigning friction on Apple Silicon.
3. **Server owns the low-latency MIDI path.** The Python side (mido /
   python-rtmidi) enumerates Core MIDI devices and streams the real-time engine's
   output to a chosen port (including a server-created virtual port a DAW or
   Pianoteq can receive from). The browser only drives the transport.
4. **Inspectable + scriptable.** Every action is a documented JSON endpoint, so
   the same backend is drivable headless for batch demos and tests.

---

## Setup (macOS / Apple Silicon)

The fastest path is the launcher, which creates the venv, installs deps, links or
fetches weights, and serves the UI:

```bash
git clone git@github.com:napanto/aria-realtime-studio.git
cd aria-realtime-studio
./run_gui.sh                 # http://localhost:8000
PORT=8800 ./run_gui.sh       # custom port
SKIP_OPEN=1 ./run_gui.sh     # don't auto-open the browser
```

Manual install (equivalent):

```bash
python3.13 -m venv .venv && source .venv/bin/activate
# Core (FastAPI/uvicorn, mido) + the Apple-Silicon MLX real-time path
# (mlx + the EleutherAI aria / ariautils packages):
pip install -e ".[mlx]"
```

> Python 3.13 is pinned because `mlx` ships wheels only up to cp313. The GUI needs
> no `torch`; it loads pre-converted MLX weights. The GUI/registry layer imports
> cleanly on non-Apple-Silicon machines — only the real-time *start* path needs
> MLX/aria and raises a clear error elsewhere.

#### Environment variables

No machine-specific paths are baked into the code. Nothing is required to run the
GUI — these are all optional:

| variable | used by | meaning |
|----------|---------|---------|
| `HF_TOKEN`  | `scripts/download_models.py` | Hugging Face access token for weight download. The repos are mostly public, so it is only needed for gated/rate-limited fetches. |
| `ARIA_REPO` | real-time engine (fallback) | path to a local EleutherAI/aria checkout, used **only** if the `aria` package is not pip-installed. |
| `LATENT_STUDIO_WEIGHTS` | weights resolution | override the `weights/` directory (e.g. to a bigger volume). |
| `ARIA_RT_NO_GRAMMAR` | real-time decoder | set to `1` to disable the grammar FSM (structural-only masking) without a redeploy. |

### Download weights

Weights are **never committed**. The launcher fetches them automatically; to do it
by hand:

```bash
export HF_TOKEN=hf_xxx        # optional; read from env, never hardcoded
python scripts/download_models.py --only aria_base aria_jazz
python scripts/download_models.py --list     # show status
```

A bare `python scripts/download_models.py` fetches **every** entry in the
registry. Weights land in `weights/<model_key>/` (gitignored). Sources:

- `aria_base` ← `loubb/aria-medium-base :: model-demo.safetensors`
- `aria_jazz` ← `napanto/jazz-piano-performance-modeling :: aria-real-time/mlx-deployed/{model,config,tokenizer-config}`

### Run

```bash
python -m gui.app          # http://127.0.0.1:8000
```

Pick a model → **Load & Start**. Choose a Core MIDI input and output, play your
keyboard, and hit **AI takeover** to hand the turn to the model; **Reset** clears
the captured context. Create a server-side virtual output to route the stream into
Pianoteq or a DAW.

---

## Status and limitations

The app has been deployed to a macOS / Apple-Silicon host and exercised
end-to-end on an M1: real-time note-by-note generation and grammar-constrained
decoding run on the device.

- **Apple-Silicon only.** The real-time path requires `mlx`, `aria`, and
  `ariautils`, which are Apple-Silicon-only. The real-time start endpoint
  instantiates the MLX engine and therefore raises a clear error on any other OS.
  The registry / download / GUI-server layers import cleanly anywhere.
- **The authoritative output is the Python engine** streaming to a Core MIDI
  device (optionally a server-held virtual output), not the browser.

---

## Coding assistants

Parts of this codebase were developed with the help of LLM-based coding
assistants, used under the direction and review of the author.

## License

Apache License 2.0 (see `LICENSE`).

`realtime/aria_demo_mlx.py` is vendored from
[EleutherAI/aria](https://github.com/EleutherAI/aria) (Apache-2.0); the
real-time engine wraps it. The jazz model weights it downloads are released
separately under non-commercial terms (CC-BY-NC-SA-4.0).
