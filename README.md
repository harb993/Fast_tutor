# Fast Tutor

Offline, voice-powered math tutor for kids. Three AI models run as HTTP microservices on localhost, orchestrated by async Python with sentence-level streaming for real-time feedback.

## Architecture

```
                       +------------------+
                       |   Browser / UI   |
                       |   (index.html)   |
                       +--------+---------+
                                |
                         HTTP requests
                                |
         +----------------------+----------------------+
         |                      |                      |
         v                      v                      v
+--------+--------+    +--------+--------+    +--------+--------+
| STT              |    | LLM              |    | TTS              |
| Moonshine Tiny   |    | Qwen 2.5-0.5B    |    | Pocket-TTS       |
| (in-process)     |    | Ollama server    |    | start-tts.sh     |
| moonshine-onnx   |    | port 11434       |    | port 8000        |
| ~26 MB           |    | ~400 MB (Q4)     |    | CPU only         |
+------------------+    +------------------+    +------------------+
```

## Data Flow

```
User speaks into mic
        |
        v
+-------+--------+
|  Voice Activity |  (numpy RMS threshold)
|  Detection      |
+-------+--------+
        |  audio chunk
        v
+-------+--------+
|  Moonshine STT  |  ~80-150ms
|  (in-process)   |
+-------+--------+
        |  transcript text
        v
+-------+--------+
| Context Builder |  merge transcript + page state -> prompt
+-------+--------+
        |  messages[]
        v
+-------+--------+
|  Qwen 2.5-0.5B |  ~50-300ms to first token
|  llama.cpp SSE  |  streams tokens via Server-Sent Events
+-------+--------+
        |  sentence boundary detected (. ? !)
        v
+-------+--------+
|  Pocket-TTS     |  ~100-200ms per sentence
|  HTTP POST      |  returns WAV audio bytes
+-------+--------+
        |  audio
        v
+-------+--------+
|  Speaker / UI   |  child hears first sentence while
|  Playback       |  LLM is still generating the rest
+--+--------------+
```

The key optimization is **sentence-level streaming**: each sentence is sent to TTS the moment it ends, without waiting for the full LLM response. This cuts perceived latency roughly in half.

## Models

### 1. Moonshine Tiny (STT)

Speech-to-text. Runs inside the Python orchestrator process via ONNX Runtime.

- Size: ~26 MB
- Runs on: CPU
- Latency: ~80-150ms per utterance
- Install: `pip install moonshine-onnx` (downloads model automatically on first use)

### 2. Qwen 2.5-0.5B Instruct (LLM)

Language model for generating tutor responses. Runs as an HTTP server via Ollama.

- Size: ~400 MB (4-bit quantization)
- Runs on: CPU or GPU
- Latency: ~50-100ms first token (GPU), ~200-500ms (CPU)

**Install and Run via Ollama:**

```bash
# Pull the model
ollama pull qwen2.5:0.5b
```

### 3. Pocket-TTS (TTS)

Text-to-speech. Runs as its own HTTP server via a shell script located in the main `Math-Tutor` repository.

- Size: downloads automatically (~200 MB first run)
- Runs on: CPU only (no GPU benefit)
- Latency: ~100-200ms per sentence

Already handled by `/home/harb/Pro/Math-Tutor/start-tts.sh`.

## Installation

### Prerequisites

- Conda environment: `ai-agent`
- Ollama installed locally
- The standard `Math-Tutor` folder at `/home/harb/Pro/Math-Tutor` for TTS

### Step-by-step

```bash
# 1. Activate conda environment
conda activate ai-agent

# 2. Install Python dependencies
pip install moonshine-onnx httpx sounddevice soundfile numpy

# 3. Pull the LLM model
ollama pull qwen2.5:0.5b
```

## Running

### Standard Mode

```bash
./start_pipeline.sh
```

### Using Procfile (alternative)

```bash
pip install honcho
honcho start
```

This starts all three services cleanly:

```
Terminal 1: /home/harb/Pro/Math-Tutor/start-tts.sh
Terminal 2: OLLAMA_ORIGINS="*" ollama serve
Terminal 3: /home/harb/miniconda3/envs/ai-agent/bin/python orchestrator.py
```

### Dashboard

Open `index.html` in a browser. Works in two modes:

- **Standalone**: Math problems work without any backend
- **Connected**: When backend is running, the pipeline panel shows real-time model activity and the tutor speaks feedback after each answer

## Files

```
Fast_tutor/
    index.html           Dashboard UI with math problems + pipeline monitor
    test_page.html       Raw connection tester for LLM and TTS endpoints
    orchestrator.py      Async pipeline: mic -> STT -> LLM -> TTS -> speaker
    start_pipeline.sh    Launches all 3 services (CPU or GPU mode)
    Procfile             Alternative startup for honcho / overmind
    README.md            This file
```

## Performance

### Expected latency (NVIDIA 6GB GPU)

```
Stage                 Time
---------------------------------
VAD phrase detection  ~50ms
Moonshine STT         ~80-150ms
LLM first token       ~50-100ms
LLM full sentence     ~200-400ms
TTS audio generation  ~100-200ms
Audio playback start  ~10ms
---------------------------------
Total to first sound  ~300-500ms
```

### Expected latency (CPU only)

```
Stage                 Time
---------------------------------
VAD phrase detection  ~50ms
Moonshine STT         ~80-150ms
LLM first token       ~200-500ms
LLM full sentence     ~500-1500ms
TTS audio generation  ~150-300ms
Audio playback start  ~10ms
---------------------------------
Total to first sound  ~800-1500ms
```

### Optimization tips

1. Use `--n-gpu-layers 99` to offload all LLM layers to GPU
2. Keep `max_tokens` low (40) for short tutor responses
3. Reduce context size (`-c 2048`) if prompts are short
4. Use `--mlock` flag to lock model in RAM and prevent swapping
5. Sentence-level streaming is already implemented -- first sentence plays while rest generates

## License

See LICENSE file.
