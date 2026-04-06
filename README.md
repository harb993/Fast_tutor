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
| Moonshine Tiny   |    | Qwen 3.5-2B      |    | Pocket-TTS       |
| (in-process)     |    | llama.cpp server  |    | uvx serve        |
| moonshine-onnx   |    | port 8080         |    | port 8000        |
| ~26 MB           |    | ~1.5 GB (Q4)     |    | CPU only         |
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
|  Qwen 3.5-2B   |  ~50-300ms to first token
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

### 2. Qwen 3.5-2B Instruct (LLM)

Language model for generating tutor responses. Runs as an HTTP server via llama.cpp.

- Size: ~1.5 GB (Q4_K_M quantization)
- Runs on: CPU or GPU (CUDA/Vulkan)
- Latency: ~50-100ms first token (GPU), ~200-500ms (CPU)

**Download the GGUF file:**

```bash
# Option A: Using huggingface-cli
pip install huggingface-hub
huggingface-cli download Qwen/Qwen3.5-2B-Instruct-GGUF \
  qwen3.5-2b-instruct-q4_k_m.gguf \
  --local-dir .

# Option B: Direct download
# Go to https://huggingface.co/Qwen/Qwen3.5-2B-Instruct-GGUF
# Download qwen3.5-2b-instruct-q4_k_m.gguf and place in project root
```

**Build llama.cpp:**

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# CPU only
cmake -B build
cmake --build build --config Release -j $(nproc)

# With NVIDIA GPU (recommended)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)

# With Intel/AMD GPU (Vulkan)
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j $(nproc)
```

Copy (or symlink) the built binary:

```bash
cp llama.cpp/build/bin/llama-server ./
```

### 3. Pocket-TTS (TTS)

Text-to-speech. Runs as its own HTTP server via uvx (zero pip install needed).

- Size: downloads automatically (~200 MB first run)
- Runs on: CPU only (no GPU benefit)
- Latency: ~100-200ms per sentence

**Install uv (required for uvx):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

No other setup needed. `uvx pocket-tts serve` handles everything.

## Installation

### Prerequisites

- Python 3.10+
- uv (for pocket-tts)
- cmake + build tools (for llama.cpp)
- NVIDIA CUDA toolkit (optional, for GPU acceleration)

### Step-by-step

```bash
# 1. Clone the repo
git clone https://github.com/harb993/Fast_tutor.git
cd Fast_tutor

# 2. Install Python dependencies
pip install moonshine-onnx httpx sounddevice soundfile numpy

# 3. Install uv (for pocket-tts)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 4. Build llama.cpp (see Models section above for GPU options)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build && cmake --build build --config Release -j $(nproc)
cp build/bin/llama-server ../
cd ..

# 5. Download the LLM model
huggingface-cli download Qwen/Qwen3.5-2B-Instruct-GGUF \
  qwen3.5-2b-instruct-q4_k_m.gguf --local-dir .
```

## Running

### CPU mode

```bash
./start_pipeline.sh
```

### GPU mode (NVIDIA)

```bash
USE_GPU=1 ./start_pipeline.sh
```

### Using Procfile (alternative)

```bash
pip install honcho
honcho start
```

This starts all three services:

```
Terminal 1: uvx pocket-tts serve --voice "hf://kyutai/tts-voices/jessica-jian/casual.wav"
Terminal 2: ./llama-server -m qwen3.5-2b-instruct-q4_k_m.gguf --port 8080 -c 2048 --threads 4
Terminal 3: python orchestrator.py
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
