#!/usr/bin/env bash
# Start the Pocket TTS server for Math Tutor voice instructions.
# Usage: ./start-tts.sh
#
# Uses uv (https://docs.astral.sh/uv/) for fast dependency management.
# First run downloads the model weights (~200MB).
# The server runs on http://localhost:8000

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.tts-venv"
UV_BIN="${UV_BIN:-uv}"

echo ""
echo "  🔊  Starting Pocket TTS server on http://localhost:8000"
echo "  📖  First run will download model + dependencies"
echo "  ⏹   Press Ctrl+C to stop"
echo ""

# ── Ensure venv exists ──────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "  📦  Creating virtual environment with uv..."
    $UV_BIN venv "$VENV_DIR" --python 3.12
fi

# ── Ensure pocket-tts is installed ──────────────────────────────
if [ ! -f "$VENV_DIR/bin/pocket-tts" ]; then
    echo "  📦  Installing pocket-tts into venv..."
    $UV_BIN pip install pocket-tts --python "$VENV_DIR/bin/python"
fi

# ── Launch the server ───────────────────────────────────────────
exec "$VENV_DIR/bin/pocket-tts" serve \
    --host 0.0.0.0 \
    --port 8000 \
    --voice azelma
