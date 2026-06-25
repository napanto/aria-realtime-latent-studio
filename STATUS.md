# STATUS — honest wiring report

> **UPDATE 2026-06-25 — MLX latent engine landed + verified on a real M1.**
> The `mlx-latent-engine` branch adds a parity-checked MLX latent path and closes
> three of the TODOs below. Verified on a MacBook Air M1/8 GB (host `hermes`):
> - **#6 (real-time MLX latent)** ✅ `latent/aria_vae_mlx_backend.py` — the z-prefix
>   (8 soft tokens) is prefilled into the KV cache + a per-layer z-residual is added
>   each step on the frozen jazz Aria MLX decoder. MLX-vs-torch parity: μ max|Δ|
>   2.4e-7, decode logits **100 % argmax**; **52 tok/s / 19 ms/token** int8. No more
>   torch full-reeval.
> - **#5 (Cadenza Performer)** ✅ the Performer was **recreated** (PiJAMA-only,
>   val ppl 37.1, `vae_campaign/A05_kongFT/performer_recreated/`) and converted to
>   MLX; `latent/cadenza_mlx_backend.py` runs the full two-stage render (Composer
>   parity 100 %, Performer fill 100 %; latent control note_density **+0.94**).
> - **#7 (probe quality)** ✅ both probes fit on **320** PiJAMA windows (mean R²
>   ~0.84–0.85), shipped as `latent_directions*.npz` in the MLX weights dirs.
>
> New registry keys `aria_vae_mlx` / `cadenza_vae_mlx` (`Backend.MLX_VAE`) select
> these; the torch/MPS backends remain as the cross-platform fallback. The engine
> modules are vendored under `studio/mlx_vae/` (byte-identical to the parity-checked
> originals). Remaining open items below: GUI takeover hook (#3), MPS dtype pass
> (#4), seed library (#8).



**This app was built and CPU-smoke-tested on Linux. It has NOT been run on
macOS / Apple Silicon.** The real-time MLX path and the MPS path are written
against the proven upstream code + our real checkpoints, but the end-to-end
macOS run (Core MIDI in/out, MLX warm-up, MPS autocast) is untested. Below is
exactly what is wired vs stubbed, plus the CPU smoke results that DO hold.

## Legend
- ✅ **wired & verified** (ran here, on CPU, against the real checkpoints)
- 🟡 **wired, unverified** (real code path, but needs macOS / a GPU / a port)
- 🟥 **stubbed / blocked** (placeholder or external dependency missing)

---

## Model abstraction (`models/`)
- ✅ Registry of all 4 backends with confirmed HF coordinates; download status
  query (`scripts/download_models.py --list`).
- ✅ `scripts/download_models.py` fetches the real files (Cadenza Composer +
  AriaVAE checkpoints downloaded and loaded here; the two MLX safetensors are
  the same well-known public files).

## Real-time MLX engine (`realtime/`) — plain Aria models
- ✅ `aria_demo_mlx.py` is the upstream EleutherAI/aria real-time demo copied
  **verbatim** — the proven low-latency engine (KV-cache, chunked prefill,
  duration recalculation, beam-of-3 first onset, min-p sampling, MIDI
  scheduling). Unmodified, so it behaves exactly as the shipped demo.
- 🟡 `RealtimeAriaEngine` (`engine.py`) — thin wrapper that points the demo at a
  chosen checkpoint + tokenizer and runs its `run()` loop in a thread. The
  wiring is straightforward, but it **requires `mlx` + `aria` + `ariautils`,
  which are Apple-Silicon-only** — it cannot import or run on this Linux box, so
  it is unverified. `aria_base` and `aria_jazz` differ only by checkpoint +
  tokenizer config (both `medium-emb`), exactly as the registry encodes.
- 🟥 `RealtimeAriaEngine.trigger_ai_takeover()` is a placeholder: the demo
  starts generating on an Enter keypress or a MIDI control CC. Driving that
  purely from the GUI button needs a small hook into the demo's control queue
  (TODO below).

## Latent core (`latent/`) — VAE models
- ✅ **AriaVAE end-to-end verified on CPU** against the real B05 checkpoint
  (`vae_campaign/B05_pipeline/stageB/last.pt`, step 39000):
  1. state-dict loads clean (missing=0, unexpected=0, full 612M frozen decoder),
  2. `encode(seed)` → `z` (shape (128,), |z|≈1.97),
  3. ridge probe fits; held-out R² on a tiny 20-window CPU sample —
     `velocity_std` +0.91, `note_density` +0.76 strongly steerable, others
     noisier (the headline figure is R²=0.908 over 400 windows in the research
     repo; 20 CPU windows is not enough to reproduce it),
  4. `z' = z + 1.5·ŵ_velocity_mean` → grammar-constrained decode →
     **valid 1230-byte MIDI continuation**.
  So load → encode → manipulate → decode is real and working.
- ✅ **Cadenza load + encode + manipulate + decode-runs verified on CPU**
  against the real A05_kongFT Composer (`vae_campaign/A05_kongFT/stageB/best.pt`):
  Composer loads (vocab 155, z_dim 128), `encode(seed)` → z (|z|≈10.9),
  `z + 1.5·dir` decodes without error.
- 🟥 **Cadenza note rendering needs the Performer checkpoint, which is NOT
  published on HF.** The Composer emits a *composition skeleton* (Pitch +
  Duration, no Velocity/MicroTiming/Pedal); the two-stage pipeline renders
  audible notes only after the Performer fills the MASK slots. The
  composition-only `decode` fallback runs but produces near-empty MIDI (the
  PerTok composition decoder needs bar/position context that the dropped-Bar
  vocab can't supply). **The Cadenza latent probe therefore can't be built from
  output attributes until a Performer `.pt` is supplied** via
  `CadenzaVAEBackend(performer_ckpt=...)`. The math/encode/inject path is
  proven; only the final render is blocked.
- ✅ `latent/probe.py` ridge probe is the exact closed-form recipe from
  `aria_vae_latent_health.py` (columns = directions); fits + caches to
  `weights/<m>/probe.npz`.
- 🟡 The AriaVAE decode uses *full-sequence re-eval per token* (no KV cache),
  matching the research generator. On CPU it is slow (~seconds/clip); on MPS it
  will be faster but is still **not** the real-time MLX path. See "MLX latent"
  TODO. This is why the README frames the VAE path as on-demand regeneration,
  not live streaming.

## GUI (`gui/`)
- ✅ FastAPI app imports cleanly; all endpoints defined (models, midi ports,
  realtime start/stop, latent load/encode/generate, attributes).
- 🟡 `gui/static/index.html`: model selector, per-attribute sliders (with probe
  R²), transport, server-MIDI device selectors, seed picker, and a built-in
  minimal SMF parser that plays the generated MIDI via WebMIDI. Written but
  **not opened in a browser here** — needs a manual macOS pass.
- 🟡 The realtime-start endpoint will only succeed on Apple Silicon (it
  instantiates the MLX engine). On any other OS it raises a clear error.
- 🟥 Browser SMF playback timing is approximate (setTimeout scheduling); fine
  for auditioning slider moves, not sample-accurate. The authoritative
  low-latency output for plain Aria is the Python engine → Core MIDI, not the
  browser.

---

## Concrete TODOs to finish on a Mac

1. **Install + import check on Apple Silicon**: `pip install -r requirements.txt`
   + the `mlx` extra; confirm `import mlx`, `import aria`, `import ariautils`,
   and `torch.backends.mps.is_available()`.
2. **Real-time MLX run**: `RealtimeAriaEngine(aria_jazz).load().start()` with a
   real Core MIDI keyboard + synth; verify warm-up compiles and latency matches
   the upstream demo. Repeat for `aria_base`.
3. **GUI takeover hook**: wire the "Generate"/transport button for plain-Aria to
   the demo's control queue (replace `trigger_ai_takeover` stub) so the AI can
   be started/stopped from the browser, not only via Enter / MIDI CC.
4. **MPS dtype pass**: confirm the VAE backends run under MPS (we force fp32 in
   autocast because bf16-on-MPS is version-flaky; try enabling fp16 autocast and
   measure).
5. **Cadenza Performer**: publish / supply the Performer checkpoint, point
   `CadenzaVAEBackend(performer_ckpt=...)` at it, then build the Cadenza probe
   (`scripts/build_probe.py --model cadenza_vae --performer-ckpt ...`). Until
   then Cadenza is encode/manipulate-only (no audible render).
6. **MLX latent path (optimization)**: the AriaVAE decoder *is* the real-time
   jazz Aria. A future optimization is to inject the z-prefix soft tokens into
   the MLX engine's KV cache so latent manipulation runs at real-time MLX speed
   instead of torch full-reeval. Today the latent path is torch/MPS on-demand;
   this is a documented, deliberate scope cut, not a bug.
7. **Probe quality**: rebuild both probes with 200–400 piano-MIDI windows
   (the smoke used ~20) to reach the research-repo R² (~0.9 for AriaVAE).
8. **Seed library**: drop a handful of jazz piano `.mid` files into
   `assets/seed_midi/` for the GUI seed picker (none are committed).
