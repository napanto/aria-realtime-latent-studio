#!/bin/bash
# Launch the browser GUI (FastAPI) for the Aria Realtime Latent Studio.
# macOS / Apple Silicon. Idempotent: sets up the venv + weights on first run,
# then just launches on subsequent runs.
#
#   ./run_gui.sh              # http://localhost:8000
#   PORT=8800 ./run_gui.sh    # custom port
#   SKIP_OPEN=1 ./run_gui.sh  # don't auto-open the browser
#
# Weights: reuses the MLX safetensors already present in a sibling
# ~/aria-realtime-latent-studio/weights (symlinked, no re-download); anything
# still missing is fetched from the Hub by scripts/download_models.py.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8000}"

command -v uv >/dev/null || { echo "uv not found (brew install uv)"; exit 1; }

# 1) venv + deps (one-time; cheap to re-run)
if [ ! -d .venv ]; then
    echo "[gui] creating venv + installing deps (one-time; pulls torch/mlx/aria)…"
    uv venv
fi
uv pip install -q -e ".[mlx]" miditok symusic pretty_midi

# 2) weights: reuse the sibling latent-studio weights if present, else download
SIB="$HOME/aria-realtime-latent-studio/weights"
mkdir -p weights/aria_vae_mlx weights/cadenza_vae_mlx
_link() { [ -f "$1" ] && [ ! -e "$2" ] && ln -s "$1" "$2" || true; }
if [ -d "$SIB/mlx" ]; then
    for f in aria_vae_decoder.safetensors aria_vae_latent.safetensors \
             aria_vae_config.json latent_directions.npz; do
        _link "$SIB/mlx/$f" "weights/aria_vae_mlx/$f"
    done
fi
if [ -d "$SIB/cadenza_mlx" ]; then
    for f in cadenza_composer.safetensors cadenza_performer.safetensors \
             cadenza_config.json latent_directions_cadenza.npz; do
        _link "$SIB/cadenza_mlx/$f" "weights/cadenza_vae_mlx/$f"
    done
fi
# Fetch anything still missing (no-op if the symlinks above satisfy the registry).
uv run python scripts/download_models.py --only aria_vae_mlx cadenza_vae_mlx || \
    echo "[gui] (download step skipped/failed — continuing with whatever is present)"

# 3) launch + open browser
echo "[gui] serving at http://localhost:$PORT  (Ctrl-C to stop)"
if [ "${SKIP_OPEN:-0}" != "1" ] && command -v open >/dev/null; then
    ( sleep 2; open "http://localhost:$PORT" ) &
fi
exec uv run uvicorn gui.app:app --host 127.0.0.1 --port "$PORT"
