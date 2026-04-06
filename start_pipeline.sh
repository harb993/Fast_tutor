#!/bin/bash
# Startup sequence for the Fast_tutor offline pipeline
# GPU MODE: Set USE_GPU=1 to enable NVIDIA CUDA acceleration
# CPU MODE: Set USE_GPU=0 (default) for CPU-only

USE_GPU=${USE_GPU:-0}

# Cleanup on exit
cleanup() {
    echo "Shutting down pipeline..."
    kill $TTS_PID $LLM_PID 2>/dev/null
    wait $TTS_PID $LLM_PID 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

echo "Starting TTS Server (pocket-tts on port 8000)..."
uvx pocket-tts serve --voice "hf://kyutai/tts-voices/jessica-jian/casual.wav" &
TTS_PID=$!

echo "Starting LLM Server (llama.cpp on port 8080)..."
if [ "$USE_GPU" = "1" ]; then
    echo "  -> GPU mode enabled (NVIDIA CUDA, full offload)"
    ./llama-server \
        -m qwen3.5-0.5b-instruct-q4_k_m.gguf \
        --port 8080 \
        -c 2048 \
        --threads 4 \
        -ngl 99 &
else
    echo "  -> CPU mode"
    ./llama-server \
        -m qwen3.5-0.5b-instruct-q4_k_m.gguf \
        --port 8080 \
        -c 2048 \
        --threads 4 &
fi
LLM_PID=$!

echo "Waiting for servers to initialize..."
sleep 5

echo "Starting Python Orchestrator..."
python orchestrator.py
