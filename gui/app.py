"""FastAPI backend for the real-time Aria studio GUI.

Endpoints
---------
GET  /                          -> the single-page UI (static/index.html)
GET  /api/models                -> registry + download status + per-model sampling
GET  /api/midi_ports            -> mido input/output port names (server side)
POST /api/realtime/start        -> load+start a plain-Aria MLX engine
POST /api/realtime/takeover     -> hand the turn to the model (start generating)
POST /api/realtime/reset        -> soft reset: clear context, keep listening
GET  /api/realtime/status       -> running flag + stream lag (ms)
POST /api/realtime/stop         -> stop the running real-time engine

This server intentionally holds at most ONE realtime engine at a time (a laptop
runs one model interactively). The heavy real-time loop runs in its own thread
inside the engine.

Run:
    python -m gui.app                # or: uvicorn gui.app:app --reload
    open http://127.0.0.1:8000
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
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

app = FastAPI(title="Aria Realtime Studio")

# ---- in-process state (single active realtime engine) ----------------------
_STATE: dict[str, object] = {
    "realtime": None,    # RealtimeAriaEngine
    "virtual_outputs": {},  # name -> open mido virtual output port (server-owned)
}


# ---- request models --------------------------------------------------------
class RealtimeStart(BaseModel):
    model_key: str
    midi_out: str
    # Optional server-held *virtual* output to stream Takeover through (the same
    # one the file-generate "play through virtual device" checkbox uses). When
    # set it overrides midi_out: the engine sends through this persistent source
    # so realtime and file playback share one device Pianoteq stays bound to.
    # Auto-created+held if it doesn't exist yet.
    output_port: Optional[str] = None
    midi_in: Optional[str] = None
    temperature: float = 0.95
    min_p: float = 0.03
    # Optional: feed a seed .mid instead of (or before) a live keyboard, so the
    # engine can be exercised without hardware. Resolved under assets/seed_midi.
    midi_path: Optional[str] = None
    # Turn control. A takeover/reset CC is wired so the GUI transport's
    # "AI takeover" button works; defaults keep plain-Aria streaming identical.
    midi_control_signal: int = 102
    midi_reset_control_signal: int = 103
    back_and_forth: bool = True
    # Turn-switch timing: None keeps the demo default (catchup, best for live
    # keyboard). "snap" suits file-seeded play (rest-then-forward streaming).
    turn_switch_mode: Optional[str] = None


class VirtualOutput(BaseModel):
    name: str


# ---- static / index --------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    idx = STATIC_DIR / "index.html"
    if not idx.exists():
        return HTMLResponse("<h1>UI missing</h1><p>gui/static/index.html not found.</p>")
    # No-cache so the browser never serves a stale UI after a redeploy (the JS is
    # inlined, so a cached page = cached JS = "my fix isn't showing up").
    return HTMLResponse(idx.read_text(), headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache", "Expires": "0",
    })


# ---- registry / devices ----------------------------------------------------
@app.get("/api/models")
def api_models() -> JSONResponse:
    out = []
    for key, spec in MODEL_REGISTRY.items():
        # Every registered model is a plain-Aria MLX model exposed in the studio.
        if spec.backend is not Backend.MLX:
            continue
        out.append(
            {
                "key": key,
                "display_name": spec.display_name,
                "backend": spec.backend.value,
                "downloaded": spec.is_downloaded(),
                "notes": spec.notes,
                "sampling": get_sampling(key),
            }
        )
    return JSONResponse(out)


class ModelDownload(BaseModel):
    key: str


@app.post("/api/models/download")
def api_models_download(req: ModelDownload) -> JSONResponse:
    """Fetch a model's weights from the Hub on a background thread. Uses ambient
    HF auth (an HF_TOKEN env var or a stored `hf` login) for gated repos."""
    import subprocess
    import sys
    import threading

    spec = get_spec(req.key)
    if spec.is_downloaded():
        return JSONResponse({"ok": True, "status": "done"})
    dls = _STATE.setdefault("downloads", {})  # type: ignore[union-attr]
    if str(dls.get(req.key, "")).startswith("running"):
        return JSONResponse({"ok": True, "status": dls[req.key]})

    def _run():
        try:
            p = subprocess.run(
                [sys.executable, "scripts/download_models.py", "--only", req.key],
                cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=3600)
            ok = get_spec(req.key).is_downloaded()
            dls[req.key] = "done" if ok else ("error: " + (p.stderr or p.stdout or "?").strip()[-220:])
        except Exception as e:  # noqa: BLE001
            dls[req.key] = f"error: {e}"

    dls[req.key] = "running…"
    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"ok": True, "status": "running"})


@app.get("/api/models/download_status")
def api_models_download_status() -> JSONResponse:
    return JSONResponse(_STATE.get("downloads", {}))  # type: ignore[arg-type]


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
def _get_or_create_virtual_output(name: str):
    """Return the server-held virtual output named ``name``, creating+holding it
    if it doesn't exist yet. One persistent CoreMIDI source per name, used by the
    realtime stream so downstream apps (Pianoteq) stay connected across turns."""
    import mido

    vo = _STATE["virtual_outputs"]  # type: ignore[assignment]
    if name not in vo:
        vo[name] = mido.open_output(name, virtual=True)
        print(f"[virtual_output] created+held '{name}'", flush=True)
    return vo[name]


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


# ---- realtime (plain Aria, MLX) -------------------------------------------
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
    if spec.backend is not Backend.MLX:
        raise HTTPException(400, f"{req.model_key} is not an MLX realtime model")
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

    # Resolve the output: a server-held virtual output (output_port) overrides
    # the plain midi_out and is injected into the engine so Takeover streams
    # through the SAME persistent source the file path uses. A real Core MIDI
    # port name is just used by name (no injection needed).
    import mido
    out_name = req.midi_out
    external_port = None
    if req.output_port:
        out_name = req.output_port
        if req.output_port not in mido.get_output_names():
            external_port = _get_or_create_virtual_output(req.output_port)

    from realtime import RealtimeAriaEngine, RealtimeConfig
    cfg = RealtimeConfig(
        checkpoint=str(spec.primary_weight),
        tokenizer_config=tok_cfg,
        midi_out=out_name,
        external_out_port=external_port,
        midi_in=req.midi_in,
        midi_path=midi_path,
        # When seeding from a file (no live keyboard), echo it through the output
        # port: the demo's file player needs a valid through-port to open (it
        # also feeds the capture queue from there), and a None port can't be
        # opened. The echo is muted while the model is generating.
        midi_through=out_name if midi_path else None,
        temperature=req.temperature,
        min_p=req.min_p,
        midi_control_signal=req.midi_control_signal,
        midi_reset_control_signal=req.midi_reset_control_signal,
        back_and_forth=req.back_and_forth,
        # File-seeded mode (no live keyboard) defaults to "snap" so the model's
        # continuation streams forward rather than racing wall-clock; a live
        # keyboard keeps the demo default (catchup, instant on Aria's fast
        # decode).
        turn_switch_mode=req.turn_switch_mode or ("snap" if midi_path else None),
    )

    eng = RealtimeAriaEngine(cfg)
    eng.load()
    eng.start()
    _STATE["realtime"] = eng
    return JSONResponse({
        "ok": True, "model": req.model_key,
        "control_cc": req.midi_control_signal,
        "output_port": out_name, "virtual": external_port is not None,
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


@app.post("/api/realtime/reset")
def api_realtime_reset() -> JSONResponse:
    """Soft reset: clear the captured context (and any in-flight generation) and
    resume listening for notes. The engine keeps running. Same effect as the
    footswitch reset CC."""
    eng = _STATE["realtime"]
    if eng is None:
        raise HTTPException(409, "no realtime engine running")
    try:
        eng.trigger_reset()  # type: ignore[attr-defined]
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return JSONResponse({"ok": True})


@app.get("/api/realtime/status")
def api_realtime_status() -> JSONResponse:
    """Live realtime status for the GUI: whether an engine runs and how far
    behind real-time the stream is (ms) — >0 means the decode can't keep up and
    the playback timeline is being stretched."""
    eng = _STATE["realtime"]
    if eng is None:
        return JSONResponse({"running": False, "lag_ms": 0})
    try:
        lag = eng.current_lag_ms()  # type: ignore[attr-defined]
        running = eng.is_running    # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        lag, running = 0, False
    return JSONResponse({"running": bool(running), "lag_ms": round(float(lag))})


@app.post("/api/realtime/stop")
def api_realtime_stop() -> JSONResponse:
    if _STATE["realtime"] is not None:
        _STATE["realtime"].stop()  # type: ignore[attr-defined]
        _STATE["realtime"] = None
    return JSONResponse({"ok": True})


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
