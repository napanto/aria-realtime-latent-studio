# Aria Realtime Latent Studio

A macOS / Apple-Silicon **real-time piano-continuation** app that runs **four**
piano-performance models behind one UI, plus a GUI to **manipulate the VAE
latent in real time** (per-attribute sliders).

| key | model | backend | role |
|-----|-------|---------|------|
| `aria_base`   | Aria (original)         | MLX (real-time engine)     | upstream EleutherAI/aria demo weights |
| `aria_jazz`   | Aria (jazz fine-tuned)  | MLX (real-time engine)     | our jazz fine-tune, already MLX-converted |
| `aria_vae`    | AriaVAE                 | PyTorch / MPS (latent)     | frozen real-time Aria decoder + 128-d latent add-on |
| `cadenza_vae` | Cadenza VAE             | PyTorch / MPS (latent)     | from-scratch Composer (+Performer) VAE, 128-d latent |

The two **plain-Aria** models are autoregressive transformers driven by the
*proven* EleutherAI/aria real-time MLX demo (KV-cache, chunked prefill,
duration recalculation, low-latency MIDI scheduling) — vendored verbatim and
wrapped in an importable engine. The two **VAE** models run in PyTorch on MPS
and expose interpretable latent sliders: move a slider, the piece re-generates
with that performance attribute shifted.

> Built and CPU-smoke-tested on Linux; **not yet run on macOS**. See
> [`STATUS.md`](STATUS.md) for exactly what is wired vs stubbed.

---

## Architecture

```
                         ┌────────────────────────── gui/ (FastAPI + WebMIDI) ──────────┐
                         │  model selector · per-attribute sliders · transport · ports  │
                         └───────────────┬───────────────────────────┬──────────────────┘
                                         │                           │
                   plain-Aria (MLX)      │                           │   VAE (torch/MPS)
                                         ▼                           ▼
   realtime/  RealtimeAriaEngine ── wraps ──► realtime/aria_demo_mlx.py      latent/ LatentBackend
     load() · start() · stop()       (vendored, unchanged)                    encode → z
     MIDI-in → continuation → MIDI-out, low-latency, KV-cache                 z' = z + Σ α·ŵ_attr
                                                                              decode → MIDI
   models/registry.py  ── single source of truth for checkpoints / tokenizers / backends ──┐
   scripts/download_models.py ── reads the same registry, fetches weights from HF ──────────┘
```

- **`models/`** — model-abstraction layer. `registry.py` declares, per model,
  the HF coordinates, the local `weights/` path, the backend, and (for VAEs)
  `z_dim`. Both the runtime and the downloader read this one table.
- **`realtime/`** — the real-time MLX engine. `aria_demo_mlx.py` is the
  upstream demo copied verbatim; `engine.py` is a thin `RealtimeAriaEngine`
  that points it at a chosen checkpoint + tokenizer and runs its loop in a
  background thread.
- **`latent/`** — the latent-manipulation core. `LatentBackend` is the common
  interface (`encode`, `decode`, `direction`, `generate_with_offsets`);
  `aria_vae_backend.py` and `cadenza_backend.py` implement it against our real
  checkpoints; `probe.py` fits the ridge probe whose columns are the
  per-attribute directions; `attributes.py` defines the seven attributes and a
  MIDI-based extractor.
- **`gui/`** — local web app (FastAPI backend + one self-contained
  `index.html`).
- **`scripts/`** — `download_models.py`, `build_probe.py`, `latent_cli.py`,
  `smoke_test.py`.

### The four latent attributes → sliders

Seven interpretable performance attributes (order matches the AriaVAE ridge
probe): **velocity_mean, velocity_std, note_density, ioi_entropy, pitch_mean,
pitch_std, pedal_fraction**. Each slider value `α` adds `α · ŵ_attr` (the
normalised ridge-probe direction) to the latent before decoding:

```
z'  =  z  +  Σ_attr  α_attr · ŵ_attr
```

The probe's per-attribute R² (shown next to each slider) tells you how reliably
the latent actually carries that attribute — low R² ⇒ that slider is weak,
honestly surfaced rather than hidden.

---

## GUI stack choice — and why

**A local web app: FastAPI (Python) backend + a single vanilla-JS page using
the browser WebMIDI API.** Considered alternatives were native (PyQt / rumps).
Rationale:

1. **Compute stays in Python.** The models are MLX and torch-MPS — both Python.
   A web UI keeps the heavy generation in-process and only ships MIDI bytes /
   slider values over localhost. No IPC bridge to a native toolkit.
2. **Zero-build sliders + selectors.** HTML range inputs and `<select>`s need
   no toolkit, no `.app` bundling, no codesigning friction on Apple Silicon.
3. **Uniform MIDI device handling.** WebMIDI enumerates and routes Core MIDI
   devices in the browser for playback, while the Python side (mido /
   python-rtmidi) owns the *live-input → real-time engine* path. Two clean
   surfaces instead of one native toolkit straddling both.
4. **Inspectable + scriptable.** Every action is a documented JSON endpoint, so
   the same backend is drivable headless (`scripts/latent_cli.py`) for batch
   demos and tests.

The trade-off (documented in `STATUS.md`): the browser's tiny built-in SMF
player gives *approximate* playback timing; the *authoritative* low-latency
real-time path for the plain-Aria models is the Python engine streaming to a
Core MIDI output, not the browser.

---

## Setup (macOS / Apple Silicon)

```bash
git clone git@github.com:napanto/aria-realtime-latent-studio.git
cd aria-realtime-latent-studio
python3.11 -m venv .venv && source .venv/bin/activate

# Core + VAE (torch/MPS) + GUI:
pip install -r requirements.txt

# Apple-Silicon real-time MLX path (the two plain-Aria models):
pip install "mlx<=0.26" \
    "aria @ git+https://github.com/EleutherAI/aria" \
    "ariautils @ git+https://github.com/EleutherAI/aria-utils"
```

The VAE backends import model definitions from the research repo. Point
`ISPR_V2_REPO` at it (or vendor it under `vendor/ispr_v2/`):

```bash
export ISPR_V2_REPO=/path/to/ispr_v2
```

### Download weights

Weights are **never committed**. Fetch them from the Hugging Face Hub:

```bash
export HF_TOKEN=hf_xxx            # read from env; never hardcoded
python scripts/download_models.py            # all four models
python scripts/download_models.py --list     # show status
```

Weights land in `weights/<model_key>/` (gitignored). Sources:

- `aria_base`   ← `loubb/aria-medium-base :: model-demo.safetensors`
- `aria_jazz`   ← `napaalm/jazz-piano-ispr-2025-2026 :: aria-real-time/mlx-deployed/{model,config,tokenizer-config}`
- `aria_vae`    ← `napaalm/jazz-piano-performance-generation :: vae_campaign/B05_pipeline/stageB/last.pt`
- `cadenza_vae` ← `napaalm/jazz-piano-performance-generation :: vae_campaign/A05_kongFT/stageB/best.pt` (Composer; Performer not published — see STATUS)

### Build the latent probe (once per VAE)

The slider directions come from a ridge probe fit on seed pieces:

```bash
python scripts/build_probe.py --model aria_vae   --seed-dir assets/seed_midi
python scripts/build_probe.py --model cadenza_vae --seed-dir assets/seed_midi
```

### Run

```bash
python -m gui.app          # http://127.0.0.1:8000
```

Pick a model → **Load & Start**. For plain-Aria, choose a Core MIDI output and
play your keyboard. For a VAE, pick a seed piece → **Encode → z**, move the
sliders, **Generate**.

Headless:

```bash
python scripts/latent_cli.py --model aria_vae \
    --seed assets/seed_midi/foo.mid \
    --set velocity_mean=+1.5 --set note_density=-1.0 --out out.mid
```

---

## License

MIT (see `LICENSE`).
