"""FastAPI backend for the latent studio GUI.

Endpoints
---------
GET  /                          -> the single-page UI (static/index.html)
GET  /api/models                -> registry + download status + per-model attrs
GET  /api/midi_ports            -> mido input/output port names (server side)
POST /api/realtime/start        -> load+start a plain-Aria OR AriaVAE MLX engine
POST /api/realtime/takeover     -> hand the turn to the model (start generating)
POST /api/realtime/latent       -> live AriaVAE z control (per-attr deltas)
GET  /api/realtime/latent_attributes -> slider spec for the running AriaVAE
POST /api/realtime/stop         -> stop the running real-time engine
POST /api/realtime/cadenza_respond -> Cadenza 2-stage call-and-response -> .mid
POST /api/latent/load           -> load a VAE backend (AriaVAE | Cadenza)
POST /api/latent/encode         -> encode a seed MIDI -> z (returned as a list)
POST /api/latent/generate       -> z + slider offsets -> MIDI continuation file
GET  /api/latent/attributes     -> slider labels + probe R² for the loaded VAE

This server intentionally holds at most ONE realtime engine and ONE latent
backend at a time (a laptop runs one model interactively). Generation is
synchronous per request; the heavy real-time loop runs in its own thread inside
the engine.

Run:
    python -m gui.app                # or: uvicorn gui.app:app --reload
    open http://127.0.0.1:8000
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError as e:  # pragma: no cover - import-time guard
    raise SystemExit(
        "GUI needs fastapi+uvicorn: pip install -r requirements.txt"
    ) from e

from models.registry import (
    MODEL_REGISTRY, Backend, get_spec, get_sampling, resolve_asset,
)
from studio import REPO_ROOT, SEED_MIDI_DIR

STATIC_DIR = Path(__file__).resolve().parent / "static"
OUT_DIR = Path(tempfile.gettempdir()) / "latent_studio_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Aria Realtime Latent Studio")

# ---- in-process state (single active model of each kind) -------------------
_STATE: dict[str, object] = {
    "realtime": None,    # RealtimeAriaEngine
    "latent": None,      # LatentBackend
    "latent_key": None,  # which VAE is loaded
    "z": None,           # current base latent (list[float])
    "virtual_outputs": {},  # name -> open mido virtual output port (server-owned)
}


# ---- request models --------------------------------------------------------
class RealtimeStart(BaseModel):
    model_key: str
    midi_out: str
    midi_in: Optional[str] = None
    temperature: float = 0.95
    min_p: float = 0.03
    # Optional: feed a seed .mid instead of (or before) a live keyboard, so the
    # engine can be exercised without hardware. Resolved under assets/seed_midi.
    midi_path: Optional[str] = None
    # AriaVAE (mlx_vae) latent options.
    latent_seed_midi: Optional[str] = None   # style anchor for the base z
    latent_prior: bool = False               # sample z~N(0,I) instead of a seed
    # Turn control. A takeover/reset CC is wired so the GUI transport's
    # "AI takeover" button works; defaults keep plain-Aria streaming identical.
    midi_control_signal: int = 102
    midi_reset_control_signal: int = 103
    back_and_forth: bool = True
    # Turn-switch timing: None keeps the demo default (catchup, best for live
    # keyboard). "snap" suits file-seeded play (rest-then-forward streaming).
    turn_switch_mode: Optional[str] = None


class RealtimeLatent(BaseModel):
    offsets: dict[str, float] = {}   # attr name -> Δ (raw attribute units)


class CadenzaRespond(BaseModel):
    seed_midi: Optional[str] = None        # phrase to respond to (else random z)
    output_port: Optional[str] = None      # play the response out this port
    offsets: dict[str, float] = {}         # latent slider offsets (alpha units)
    temperature: float = 1.0
    top_k: int = 24
    top_p: float = 1.0
    use_random_z: bool = False


class LatentLoad(BaseModel):
    model_key: str            # "aria_vae" | "cadenza_vae"
    performer_ckpt: Optional[str] = None
    probe_path: Optional[str] = None


class LatentEncode(BaseModel):
    seed_midi: str            # path under assets/seed_midi or absolute


class LatentGenerate(BaseModel):
    offsets: dict[str, float] = {}   # attr -> alpha (slider value)
    temperature: float = 0.95
    top_p: float = 1.0               # AriaVAE nucleus (1.0 = off)
    top_k: int = 0                   # Cadenza top-k (0 = off)
    use_random_z: bool = False
    output_port: Optional[str] = None  # if set, also play the result out this port


class VirtualOutput(BaseModel):
    name: str


# ---- static / index --------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    idx = STATIC_DIR / "index.html"
    if not idx.exists():
        return "<h1>UI missing</h1><p>gui/static/index.html not found.</p>"
    return idx.read_text()


# ---- registry / devices ----------------------------------------------------
@app.get("/api/models")
def api_models() -> JSONResponse:
    out = []
    for key, spec in MODEL_REGISTRY.items():
        out.append(
            {
                "key": key,
                "display_name": spec.display_name,
                "backend": spec.backend.value,
                "downloaded": spec.is_downloaded(),
                "z_dim": spec.z_dim,
                "notes": spec.notes,
                "sampling": get_sampling(key),
            }
        )
    return JSONResponse(out)


@app.get("/api/midi_ports")
def api_midi_ports() -> JSONResponse:
    try:
        import mido

        return JSONResponse(
            {"inputs": mido.get_input_names(), "outputs": mido.get_output_names()}
        )
    except Exception as e:
        return JSONResponse({"inputs": [], "outputs": [], "error": str(e)})


@app.get("/api/midi_config")
def api_midi_config() -> JSONResponse:
    """The saved MIDI devices config (config/midi_devices.json): default
    input/output port names + control/reset CCs for the host rig. The GUI
    pre-selects the device dropdowns from this; empty {} if absent."""
    import json

    p = REPO_ROOT / "config" / "midi_devices.json"
    if not p.exists():
        return JSONResponse({})
    try:
        return JSONResponse(json.loads(p.read_text()))
    except Exception as e:
        return JSONResponse({"error": str(e)})


# ---- virtual MIDI outputs (server-created) ---------------------------------
def _play_file_through(midi_path: str, port_name: str) -> None:
    """Play a .mid file out a port (a server-owned virtual port if known, else a
    real port opened by name) on a background thread, with the file's timing."""
    import threading
    import mido

    def _run():
        vo = _STATE["virtual_outputs"]  # type: ignore[assignment]
        owned = port_name in vo
        port = None
        try:
            # open_output is inside the try: an unknown port name (not a
            # server-owned virtual port and not a real Core MIDI destination)
            # would otherwise raise here and kill this daemon thread with a
            # noisy traceback. Failing to play is non-fatal — the .mid is still
            # returned to the caller.
            port = vo[port_name] if owned else mido.open_output(port_name)
            for msg in mido.MidiFile(midi_path).play():
                port.send(msg)
        except Exception:
            pass
        finally:
            if port is not None and not owned:
                try:
                    port.close()
                except Exception:
                    pass

    threading.Thread(target=_run, daemon=True).start()


@app.post("/api/midi/create_virtual_output")
def api_create_virtual_output(req: VirtualOutput) -> JSONResponse:
    """Create a persistent virtual MIDI output port other apps (Pianoteq, a DAW)
    can receive from. Held open for the server's lifetime; the studio sends
    generated/realtime MIDI through it."""
    import mido

    vo = _STATE["virtual_outputs"]  # type: ignore[assignment]
    if req.name in vo:
        return JSONResponse({"ok": True, "name": req.name, "already": True,
                             "virtual_outputs": list(vo)})
    try:
        vo[req.name] = mido.open_output(req.name, virtual=True)
    except Exception as e:  # backend without virtual-port support
        raise HTTPException(
            400, f"could not create virtual output '{req.name}': {e} "
                 "(needs the python-rtmidi backend; macOS/Linux only)")
    return JSONResponse({"ok": True, "name": req.name, "virtual_outputs": list(vo)})


@app.post("/api/midi/close_virtual_output")
def api_close_virtual_output(req: VirtualOutput) -> JSONResponse:
    vo = _STATE["virtual_outputs"]  # type: ignore[assignment]
    port = vo.pop(req.name, None)
    if port is not None:
        try:
            port.close()
        except Exception:
            pass
    return JSONResponse({"ok": True, "virtual_outputs": list(vo)})


@app.get("/api/midi/virtual_outputs")
def api_virtual_outputs() -> JSONResponse:
    return JSONResponse({"virtual_outputs": list(_STATE["virtual_outputs"])})  # type: ignore[arg-type]


@app.get("/api/seed_midi")
def api_seed_midi() -> JSONResponse:
    files = []
    if SEED_MIDI_DIR.exists():
        files = sorted(p.name for p in SEED_MIDI_DIR.glob("*.mid"))
    return JSONResponse(files)


# ---- realtime (plain Aria + AriaVAE latent, MLX) --------------------------
def _resolve_seed(path: Optional[str]) -> Optional[str]:
    """Resolve a seed-MIDI request field to an absolute path (under
    assets/seed_midi if relative). Returns None for an empty field."""
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = SEED_MIDI_DIR / path
    if not p.exists():
        raise HTTPException(404, f"seed MIDI not found: {p}")
    return str(p)


@app.post("/api/realtime/start")
def api_realtime_start(req: RealtimeStart) -> JSONResponse:
    spec = get_spec(req.model_key)
    if spec.backend not in (Backend.MLX, Backend.MLX_VAE):
        raise HTTPException(400, f"{req.model_key} is not an MLX realtime model")
    # AriaVAE (aria_vae_mlx) streams in real time; Cadenza cannot stream
    # note-by-note (use /api/realtime/cadenza_respond instead).
    if spec.backend is Backend.MLX_VAE and req.model_key != "aria_vae_mlx":
        raise HTTPException(
            400,
            f"{req.model_key} does not stream in real time; use "
            "/api/realtime/cadenza_respond for the Cadenza phrase loop",
        )
    if not spec.is_downloaded():
        raise HTTPException(409, f"{req.model_key} weights missing; download first")

    tok_cfg = str(resolve_asset(spec.tokenizer_config_local))
    midi_path = _resolve_seed(req.midi_path)
    # Real-time turn-taking needs an input to build context from: a live MIDI
    # keyboard (midi_in) or a seed file (midi_path). Without either, capture
    # never fills and a takeover would hand the model an empty context.
    if not req.midi_in and not midi_path:
        raise HTTPException(
            400, "realtime needs a MIDI input (midi_in) or a seed file (midi_path)"
        )

    # Stop any prior engine.
    if _STATE["realtime"] is not None:
        _STATE["realtime"].stop()  # type: ignore[attr-defined]
        _STATE["realtime"] = None

    from realtime import RealtimeAriaEngine, RealtimeConfig
    cfg = RealtimeConfig(
        checkpoint=str(spec.primary_weight),
        tokenizer_config=tok_cfg,
        midi_out=req.midi_out,
        midi_in=req.midi_in,
        midi_path=midi_path,
        # When seeding from a file (no live keyboard), echo it through the output
        # port: the demo's file player needs a valid through-port to open (it
        # also feeds the capture queue from there), and a None port can't be
        # opened. The echo is muted while the model is generating.
        midi_through=req.midi_out if midi_path else None,
        temperature=req.temperature,
        min_p=req.min_p,
        midi_control_signal=req.midi_control_signal,
        midi_reset_control_signal=req.midi_reset_control_signal,
        back_and_forth=req.back_and_forth,
        # File-seeded mode (no live keyboard) defaults to "snap" so the model's
        # continuation streams forward rather than racing wall-clock; a live
        # keyboard keeps the demo default (catchup).
        turn_switch_mode=req.turn_switch_mode or ("snap" if midi_path else None),
    )
    if spec.backend is Backend.MLX_VAE:
        # AriaVAE: the VAE decoder becomes the streamed model (z-prefix +
        # per-layer z-residual). weights_subdir holds the converted MLX
        # safetensors + config + latent_directions.npz.
        cfg.latent_dir = str(spec.weights_subdir)
        cfg.latent_seed_midi = _resolve_seed(req.latent_seed_midi)
        cfg.latent_prior = req.latent_prior

    eng = RealtimeAriaEngine(cfg)
    eng.load()
    eng.start()
    _STATE["realtime"] = eng
    return JSONResponse({
        "ok": True, "model": req.model_key, "latent": eng.is_latent,
        "control_cc": req.midi_control_signal,
    })


@app.post("/api/realtime/takeover")
def api_realtime_takeover() -> JSONResponse:
    """Hand the turn over to the model now (the GUI transport button / the CLI
    control CC). Ends the current capture window and starts generation."""
    eng = _STATE["realtime"]
    if eng is None:
        raise HTTPException(409, "no realtime engine running")
    try:
        eng.trigger_ai_takeover()  # type: ignore[attr-defined]
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return JSONResponse({"ok": True})


@app.post("/api/realtime/latent")
def api_realtime_latent(req: RealtimeLatent) -> JSONResponse:
    """Live AriaVAE latent control: move z by per-attribute deltas while the
    engine streams. The run loop picks up the new residuals on the next token."""
    eng = _STATE["realtime"]
    if eng is None:
        raise HTTPException(409, "no realtime engine running")
    try:
        applied = eng.set_latent_offsets(req.offsets)  # type: ignore[attr-defined]
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return JSONResponse({"ok": True, "applied": applied})


@app.get("/api/realtime/latent_attributes")
def api_realtime_latent_attributes() -> JSONResponse:
    """Slider spec (name/label/r2) for the running AriaVAE engine; [] if the
    engine is plain Aria or not running."""
    eng = _STATE["realtime"]
    if eng is None or not getattr(eng, "is_latent", False):
        return JSONResponse({"attributes": [], "latent": False})
    return JSONResponse({
        "attributes": eng.latent_attributes(),  # type: ignore[attr-defined]
        "latent": True,
    })


@app.post("/api/realtime/stop")
def api_realtime_stop() -> JSONResponse:
    if _STATE["realtime"] is not None:
        _STATE["realtime"].stop()  # type: ignore[attr-defined]
        _STATE["realtime"] = None
    return JSONResponse({"ok": True})


# ---- Cadenza phrase call-and-response (two-stage, not note-streaming) ------
@app.post("/api/realtime/cadenza_respond")
def api_cadenza_respond(req: CadenzaRespond) -> FileResponse:
    """Respond to an input phrase with Cadenza's two-stage generation.

    Cadenza (Composer AR -> Performer bidirectional fill) cannot stream
    note-by-note, so real-time interaction is a call-and-response: encode the
    seed phrase (or a random/last z), run the 2-stage generate, write a .mid,
    and (optionally) play it out a server MIDI port via the file player. The
    response .mid is also returned so the browser can play it via WebMIDI.
    """
    import numpy as np

    backend = _STATE["latent"]
    if backend is None or _STATE["latent_key"] != "cadenza_vae_mlx":
        spec = get_spec("cadenza_vae_mlx")
        if not spec.is_downloaded():
            raise HTTPException(409, "cadenza_vae_mlx weights missing; download first")
        from latent.cadenza_mlx_backend import CadenzaVAEMLXBackend

        backend = CadenzaVAEMLXBackend(weights_dir=str(spec.weights_subdir))
        backend.load()
        _STATE["latent"] = backend
        _STATE["latent_key"] = "cadenza_vae_mlx"
        _STATE["z"] = None

    seed = _resolve_seed(req.seed_midi)
    if req.use_random_z:
        z = backend.random_z()  # type: ignore[attr-defined]
    elif seed is not None:
        z = backend.encode(seed)  # type: ignore[attr-defined]
    elif _STATE["z"] is not None:
        z = np.asarray(_STATE["z"], dtype=np.float32)
    else:
        z = backend.random_z()  # type: ignore[attr-defined]
    _STATE["z"] = z.tolist()

    out = OUT_DIR / f"cadenza_{int(time.time() * 1000)}.mid"
    backend.generate_with_offsets(  # type: ignore[attr-defined]
        z, req.offsets, str(out),
        temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
    )
    if req.output_port:
        _play_file_through(str(out), req.output_port)
    return FileResponse(str(out), media_type="audio/midi", filename=out.name)


# ---- latent (VAE) ----------------------------------------------------------
@app.post("/api/latent/load")
def api_latent_load(req: LatentLoad) -> JSONResponse:
    spec = get_spec(req.model_key)
    if spec.backend not in (Backend.TORCH_VAE, Backend.MLX_VAE):
        raise HTTPException(400, f"{req.model_key} is not a VAE latent model")
    if not spec.is_downloaded():
        raise HTTPException(409, f"{req.model_key} weights missing; download first")

    if req.model_key == "aria_vae":
        from latent.aria_vae_backend import AriaVAEBackend

        backend = AriaVAEBackend(
            checkpoint=str(spec.primary_weight),
            tokenizer_config=str(resolve_asset(spec.tokenizer_config_local)),
            probe_path=req.probe_path
            or str(spec.weights_subdir / "probe.npz"),
        )
    elif req.model_key == "cadenza_vae":
        from latent.cadenza_backend import CadenzaVAEBackend

        backend = CadenzaVAEBackend(
            composer_ckpt=str(spec.primary_weight),
            performer_ckpt=req.performer_ckpt,
            probe_path=req.probe_path
            or str(spec.weights_subdir / "probe.npz"),
        )
    elif req.model_key == "aria_vae_mlx":
        # Real-time MLX latent path (Apple Silicon). Probe directions ship in
        # the weights dir (latent_directions.npz); no separate probe build.
        from latent.aria_vae_mlx_backend import AriaVAEMLXBackend

        backend = AriaVAEMLXBackend(
            weights_dir=str(spec.weights_subdir),
            tokenizer_config=str(resolve_asset(spec.tokenizer_config_local)),
            quantize=bool(getattr(req, "quantize", False)),
        )
    elif req.model_key == "cadenza_vae_mlx":
        from latent.cadenza_mlx_backend import CadenzaVAEMLXBackend

        backend = CadenzaVAEMLXBackend(weights_dir=str(spec.weights_subdir))
    else:
        raise HTTPException(400, f"unknown VAE key {req.model_key}")

    backend.load()
    _STATE["latent"] = backend
    _STATE["latent_key"] = req.model_key
    _STATE["z"] = None
    return JSONResponse({"ok": True, "model": req.model_key, "z_dim": backend.z_dim})


@app.get("/api/latent/attributes")
def api_latent_attributes() -> JSONResponse:
    backend = _STATE["latent"]
    if backend is None:
        raise HTTPException(409, "no VAE loaded")
    from latent.attributes import ATTR_LABELS

    probe = getattr(backend, "_probe", None)
    r2 = probe.r2 if probe is not None else {}
    attrs = [
        {
            "name": a,
            "label": ATTR_LABELS.get(a, a),
            "r2": round(float(r2.get(a, 0.0)), 3) if r2 else None,
        }
        for a in backend.attribute_names  # type: ignore[attr-defined]
    ]
    return JSONResponse({"attributes": attrs, "probe_ready": probe is not None})


@app.post("/api/latent/encode")
def api_latent_encode(req: LatentEncode) -> JSONResponse:
    backend = _STATE["latent"]
    if backend is None:
        raise HTTPException(409, "no VAE loaded")
    seed = Path(req.seed_midi)
    if not seed.is_absolute():
        seed = SEED_MIDI_DIR / req.seed_midi
    if not seed.exists():
        raise HTTPException(404, f"seed MIDI not found: {seed}")
    z = backend.encode(str(seed))  # type: ignore[attr-defined]
    _STATE["z"] = z.tolist()
    return JSONResponse({"ok": True, "z_dim": int(z.shape[0])})


@app.post("/api/latent/generate")
def api_latent_generate(req: LatentGenerate) -> FileResponse:
    backend = _STATE["latent"]
    if backend is None:
        raise HTTPException(409, "no VAE loaded")
    import numpy as np

    if req.use_random_z or _STATE["z"] is None:
        z = backend.random_z()  # type: ignore[attr-defined]
        _STATE["z"] = z.tolist()
    else:
        z = np.asarray(_STATE["z"], dtype=np.float32)

    out = OUT_DIR / f"gen_{int(time.time() * 1000)}.mid"
    backend.generate_with_offsets(  # type: ignore[attr-defined]
        z,
        req.offsets,
        str(out),
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
    )
    # Optionally also play it out a (virtual or real) server-side MIDI port.
    if req.output_port:
        _play_file_through(str(out), req.output_port)
    return FileResponse(str(out), media_type="audio/midi", filename=out.name)


_STDIN_KEEPALIVE_FD = None  # write end of the headless stdin pipe (kept open)


def _detach_stdin_if_headless() -> None:
    """Headless safety for the vendored real-time demo.

    The demo's terminal control listens for Enter on ``sys.stdin`` and treats an
    empty line as "hand the turn to the AI". When the server runs detached
    (nohup/systemd) stdin is ``/dev/null``, which reports EOF (an empty line) on
    every poll — busy-firing that takeover and crashing the capture turn on an
    empty message list. The GUI is driven entirely over HTTP, so when we are NOT
    on an interactive TTY we point stdin at a pipe that never yields, making the
    listener idle. Interactive terminals keep Enter-to-takeover.
    """
    import os
    import sys

    try:
        if sys.stdin is not None and sys.stdin.isatty():
            return
    except (ValueError, OSError):
        pass  # stdin already closed -> treat as headless
    r, w = os.pipe()
    sys.stdin = os.fdopen(r)
    global _STDIN_KEEPALIVE_FD
    _STDIN_KEEPALIVE_FD = w  # keep the write end open so the read end never EOFs


def main() -> None:
    import uvicorn

    _detach_stdin_if_headless()
    uvicorn.run("gui.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
