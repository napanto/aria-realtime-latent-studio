#!/bin/bash
# Launch the browser GUI (FastAPI) for the Aria Realtime Studio.
# macOS / Apple Silicon. Idempotent: sets up the venv + weights on first run,
# then just launches on subsequent runs.
#
#   ./run_gui.sh              # http://localhost:8000
#   PORT=8800 ./run_gui.sh    # custom port
#   SKIP_OPEN=1 ./run_gui.sh  # don't auto-open the browser
#
# Weights: reuses the MLX safetensors already present in a sibling
# ~/aria-realtime-studio/weights (symlinked, no re-download); anything
# still missing is fetched from the Hub by scripts/download_models.py.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8000}"

# Ensure Homebrew tools (uv) are on PATH even in non-login shells.
for d in /opt/homebrew/bin /usr/local/bin "$HOME/.local/bin"; do
    [ -d "$d" ] && case ":$PATH:" in *":$d:"*) ;; *) PATH="$d:$PATH";; esac
done
export PATH
command -v uv >/dev/null || { echo "uv not found (brew install uv)"; exit 1; }

# 1) venv + deps (one-time; cheap to re-run).
# Pin Python 3.13 — mlx ships wheels only up to cp313 (no 3.14 yet).
PYVER="${PYVER:-3.13}"
if [ ! -x .venv/bin/python ]; then
    echo "[gui] creating venv (python $PYVER) + installing deps (one-time; pulls torch/mlx/aria)…"
    uv venv --python "$PYVER"
fi
# Install only if the venv isn't already complete (skips re-resolution on relaunch).
if ! .venv/bin/python -c "import fastapi, uvicorn, mlx.core, aria, ariautils" 2>/dev/null; then
    echo "[gui] installing deps…"
    uv pip install -e ".[mlx]" pretty_midi
fi

# 2) weights: reuse the big safetensors already in ~/aria/models if present
# (the original demo + jazz deploy), so we don't re-pull ~5 GB. The small jazz
# config/tokenizer JSONs are fetched by the download step below if still missing.
_link() { [ -f "$1" ] && [ ! -e "$2" ] && ln -s "$1" "$2" || true; }
AM="$HOME/aria/models"
mkdir -p weights/aria_base weights/aria_jazz
[ -f "$AM/model-demo.safetensors" ] && _link "$AM/model-demo.safetensors" "weights/aria_base/model-demo.safetensors"
[ -f "$AM/aria-real-time-jazz.safetensors" ] && _link "$AM/aria-real-time-jazz.safetensors" "weights/aria_jazz/model.safetensors"
# Fetch anything still missing (no-op if the symlinks above satisfy the registry).
uv run python scripts/download_models.py --only aria_base aria_jazz || \
    echo "[gui] (download step skipped/failed — continuing with whatever is present)"

# 3) launch + open browser
echo "[gui] serving at http://localhost:$PORT  (Ctrl-C to stop)"
if [ "${SKIP_OPEN:-0}" != "1" ] && command -v open >/dev/null; then
    ( sleep 2; open "http://localhost:$PORT" ) &
fi
exec uv run uvicorn gui.app:app --host 127.0.0.1 --port "$PORT"
