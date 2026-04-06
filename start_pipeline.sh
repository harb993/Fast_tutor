#!/bin/bash
# Startup sequence for the Fast_tutor offline pipeline using Ollama and Conda

# Cleanup on exit
cleanup() {
    echo "Shutting down pipeline..."
    kill $TTS_PID $LLM_PID 2>/dev/null
    wait $TTS_PID $LLM_PID 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

echo "Starting TTS Server..."
/home/harb/Pro/Math-Tutor/start-tts.sh &
TTS_PID=$!

echo "Starting LLM Server (Ollama)..."
OLLAMA_ORIGINS="*" ollama serve &
LLM_PID=$!

echo "Waiting for servers to initialize..."
sleep 3

echo "Starting Python Orchestrator using Conda ai-agent environment..."
/home/harb/miniconda3/envs/ai-agent/bin/python orchestrator.py
