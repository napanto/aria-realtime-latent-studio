#!/bin/bash
# CLI Cadenza call-and-response (GUI-repo copy of the latent-studio script).
#
# Cadenza (Composer AR -> Performer bidirectional fill) cannot stream
# note-by-note, so its "real-time" interaction is phrase-based: take an input
# phrase (a seed .mid), encode it -> z, run the two-stage generation, write a
# .mid response, and (optionally) play that response out a MIDI port. This is
# the CLI counterpart of the server's POST /api/realtime/cadenza_respond.
#
# Config via environment (all optional):
#   WEIGHTS_DIR  dir with cadenza_{composer,performer}.safetensors +
#                cadenza_config.json + latent_directions_cadenza.npz
#                                              (default weights/cadenza_vae_mlx)
#   SEED_MIDI    the phrase to respond to    (default assets/seed_midi/pokey_jazz.mid)
#   OUT_DIR      where .mid responses land            (default out/cadenza)
#   MIDI_OUT     if set, play the two-stage response out this port
#                (created as a virtual port if it does not already exist)
#   PROMPT_LEN   seed tokens to encode                (default 384)
#   MAX_STEPS    Composer generation length           (default 512)
#   TEMP / TOP_K / TOP_P / SEED                       (default 1.0 / 24 / 1.0 / 0)
#   PERFORMER_SAMPLE=1   sample the Performer fill instead of argmax
#   PY           python to use (default: .venv/bin/python if present, else python3)

cd "$(dirname "$0")"

WEIGHTS_DIR="${WEIGHTS_DIR:-weights/cadenza_vae_mlx}"
SEED_MIDI="${SEED_MIDI:-assets/seed_midi/pokey_jazz.mid}"
OUT_DIR="${OUT_DIR:-out/cadenza}"
PROMPT_LEN="${PROMPT_LEN:-384}"
MAX_STEPS="${MAX_STEPS:-512}"
TEMP="${TEMP:-1.0}"
TOP_K="${TOP_K:-24}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-0}"

if [ -z "$PY" ]; then
    if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3"; fi
fi

ARGS=(
    --weights_dir "$WEIGHTS_DIR"
    --seed_midi "$SEED_MIDI"
    --out_dir "$OUT_DIR"
    --prompt_len "$PROMPT_LEN"
    --max_generate_steps "$MAX_STEPS"
    --temperature "$TEMP"
    --top_k "$TOP_K"
    --top_p "$TOP_P"
    --seed "$SEED"
)
[ "${PERFORMER_SAMPLE:-0}" = "1" ] && ARGS+=(--performer_sample)

echo "Cadenza respond: seed=$SEED_MIDI  weights=$WEIGHTS_DIR  out=$OUT_DIR  ($PY)"
"$PY" studio/mlx_vae/cadenza_two_stage_mlx.py "${ARGS[@]}"

# Optionally play the two-stage response out a MIDI port.
if [ -n "$MIDI_OUT" ]; then
    STEM="$(basename "${SEED_MIDI%.*}")"
    RESP="$OUT_DIR/${STEM}_two_stage.mid"
    echo "Playing $RESP through '$MIDI_OUT'…"
    "$PY" - "$RESP" "$MIDI_OUT" <<'PY'
import sys, mido
path, port_name = sys.argv[1], sys.argv[2]
try:
    out = mido.open_output(port_name)
except (IOError, OSError):
    out = mido.open_output(port_name, virtual=True)
for msg in mido.MidiFile(path).play():
    out.send(msg)
out.close()
PY
fi
