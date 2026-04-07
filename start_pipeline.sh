#!/usr/bin/env bash
# Fast Tutor Unified Pipeline Launcher
# Start this script to boot all interconnected AI services locally.

# ==========================================
# ⚙️ CONFIGURATION & PORTS
# Change these securely if you have conflicts
# ==========================================
export OLLAMA_PORT="11434"
export TTS_PORT="8000"
export ORCHESTRATOR_PORT="8001"

# ==========================================
# 🧠 LOCAL MODEL WAREHOUSE
# (Commented out HF_HOME so it reuses previously downloaded models in ~/.cache/huggingface!)
# ==========================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export LOCAL_MODELS_DIR="$SCRIPT_DIR/local_models"

# export HF_HOME="$LOCAL_MODELS_DIR/hf"
export OLLAMA_MODELS="$LOCAL_MODELS_DIR/ollama"
export UV_BIN="${UV_BIN:-uv}"

mkdir -p "$OLLAMA_MODELS"

echo "========================================================="
echo " Starting Fast Tutor Unified Local Pipeline"
echo " Ports: Orchestrator [$ORCHESTRATOR_PORT] | TTS [$TTS_PORT] | Ollama [$OLLAMA_PORT]"
echo " Models: $LOCAL_MODELS_DIR"
echo "========================================================="

# ── 1. Boot Local Pocket TTS Server ──────────────────────────────
echo -e "\n[1/3] Booting Voice Engine (Pocket-TTS)..."
# We now use the directly copied .tts-venv from Math-Tutor inside Fast_tutor!
TTS_VENV="$SCRIPT_DIR/.tts-venv"

if [ ! -d "$TTS_VENV" ]; then
    echo "  📦  Creating virtual environment for TTS..."
    $UV_BIN venv "$TTS_VENV" --python 3.12
fi

if [ ! -f "$TTS_VENV/bin/pocket-tts" ]; then
    echo "  📦  Installing pocket-tts..."
    $UV_BIN pip install pocket-tts --python "$TTS_VENV/bin/python"
fi

# Run TTS server in background
"$TTS_VENV/bin/pocket-tts" serve --host 0.0.0.0 --port "$TTS_PORT" --voice azelma > /dev/null 2>&1 &
TTS_PID=$!

# ── 2. Boot Local Ollama Engine ──────────────────────────────────
echo "[2/3] Checking LLM Engine (Ollama)..."

# If Ollama is running globally, we warn, otherwise we'd start a custom instance.
# For now, we rely on the host's robust Ollama instance but configure environment.
# Since ollama runs as a system service heavily, setting OLLAMA_MODELS might require
# restarting the service. If Ollama is already running on port, we just use it.
OLLAMA_URL="http://localhost:${OLLAMA_PORT}/"
if curl -s "$OLLAMA_URL" > /dev/null; then
  echo "  ✅ Ollama is active on port $OLLAMA_PORT"
else
  # Try to spin up local
  echo "  🚀 Starting local isolated Ollama daemon..."
  ollama serve > /dev/null 2>&1 &
  OLLAMA_PID=$!
  sleep 4
fi

# ── 3. Boot Python Orchestrator Backbone ─────────────────────────
echo -e "\n[3/3] Booting Python Backbone..."
echo "  🎧 Ensure your microphone is plugged in."

eval "$(conda shell.bash hook)"
conda activate ai-agent

# Execute orchestrator in the foreground so Ctrl+C gracefully works
python orchestrator.py

# Cleanup process function
cleanup() {
    echo -e "\nShutting down pipeline components..."
    kill $TTS_PID 2>/dev/null
    kill $OLLAMA_PID 2>/dev/null
    echo "Done."
}

trap cleanup EXIT
