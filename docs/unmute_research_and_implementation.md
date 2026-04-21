# Unmute.sh Research & Implementation Documentation

> **Project**: Fast_tutor — Offline AI Math Tutor  
> **Date**: April 2026  
> **Goal**: Improve conversation naturalness by adapting techniques from Kyutai's Unmute.sh framework

---

## Table of Contents

1. [What is Unmute.sh?](#1-what-is-unmutesh)
2. [Unmute Architecture Deep Dive](#2-unmute-architecture-deep-dive)
3. [Key Source Files Analyzed](#3-key-source-files-analyzed)
4. [How Unmute Handles Pauses & Responses](#4-how-unmute-handles-pauses--responses)
5. [Gap Analysis: Fast_tutor vs Unmute](#5-gap-analysis-fast_tutor-vs-unmute)
6. [Implementation Plan](#6-implementation-plan)
7. [What Was Implemented](#7-what-was-implemented)
8. [Testing Results](#8-testing-results)

---

## 1. What is Unmute.sh?

**Unmute** is an open-source framework by [Kyutai Labs](https://kyutai.org/) that makes text-based LLMs "listen and speak" by wrapping them in low-latency STT and TTS models.

- **GitHub**: [kyutai-labs/unmute](https://github.com/kyutai-labs/unmute)
- **Demo**: [unmute.sh](https://unmute.sh)
- **License**: MIT
- **Stars**: 1.3k+

Unlike monolithic speech-to-speech models (like Kyutai's earlier "Moshi"), Unmute is a **cascaded modular system**:

```
User Mic → [STT] → Text → [LLM] → Text → [TTS] → Speaker
```

This allows swapping in any LLM (GPT, Llama, Gemma, Qwen, etc.) while using Kyutai's optimized STT and TTS for the voice layer.

---

## 2. Unmute Architecture Deep Dive

### High-Level Flow

```
┌────────────┐     WebSocket      ┌──────────────────────┐
│   Browser   │ ◄──────────────► │   Python Backend      │
│ (Next.js)   │    Opus audio     │  (unmute_handler.py)  │
└────────────┘                    └──────────┬───────────┘
                                             │
                          ┌──────────────────┼──────────────────┐
                          │                  │                  │
                    ┌─────▼─────┐   ┌───────▼──────┐   ┌──────▼──────┐
                    │ STT Server │   │  LLM Server  │   │ TTS Server  │
                    │ (Rust)     │   │ (vLLM/Ollama)│   │ (Rust)      │
                    │ WebSocket  │   │ OpenAI API   │   │ WebSocket   │
                    └───────────┘   └──────────────┘   └─────────────┘
```

### Key Services

| Service | Technology | Connection | Latency |
|---------|-----------|------------|---------|
| STT | Kyutai stt-1b (Rust) | WebSocket streaming | Real-time |
| LLM | Any OpenAI-compatible | HTTP streaming | ~200-500ms TTFT |
| TTS | Kyutai tts-1.6b (Rust) | WebSocket streaming | ~220-350ms |

### Backend Orchestrator

The core logic lives in `unmute_handler.py` — a Python class (`UnmuteHandler`) that:
1. Receives audio frames from the browser
2. Forwards them to STT via WebSocket
3. Detects pauses using Semantic VAD
4. Triggers LLM generation
5. Streams LLM output word-by-word to TTS
6. Sends TTS audio back to browser
7. Handles interruptions at every stage

---

## 3. Key Source Files Analyzed

### `unmute/unmute_handler.py` (Main orchestrator)
- **656 lines** — the core conversation handler
- Manages the `UnmuteHandler` class extending `AsyncStreamHandler`
- Contains: audio receive loop, pause detection, interruption handling, TTS/LLM coordination
- Key constants:
  - `USER_SILENCE_TIMEOUT = 7.0` — seconds before nudging silent user
  - `UNINTERRUPTIBLE_BY_VAD_TIME_SEC = 3` — grace period before allowing interruption
  - `FIRST_MESSAGE_TEMPERATURE = 0.7` / `FURTHER_MESSAGES_TEMPERATURE = 0.3`

### `unmute/llm/chatbot.py` (State machine)
- **128 lines** — conversation state management
- State is **derived from chat_history**, not an explicit variable:
  - Last message is `assistant` → `bot_speaking`
  - Last message is `user` + non-empty → `user_speaking`
  - Last message is `user` + empty → `waiting_for_user`
- Handles message delta accumulation with proper space insertion
- Preprocesses messages before LLM calls

### `unmute/llm/llm_utils.py` (LLM utilities)
- **172 lines** — LLM streaming and text processing
- `rechunk_to_words()` — re-chunks LLM token stream into whole words for TTS
- `preprocess_messages_for_llm()` — cleans chat history:
  - Removes interruption markers (em-dash `—`)
  - Merges consecutive same-role messages
  - Handles the `"..."` silence marker
- `VLLMStream` — async OpenAI client wrapper

### `unmute/openai_realtime_api_events.py` (WebSocket protocol)
- Defines the message types between frontend and backend
- Based on OpenAI Realtime API with custom extensions

---

## 4. How Unmute Handles Pauses & Responses

### 4.1 Semantic VAD (Pause Detection)

**The single most important technique.** Instead of using volume-based silence detection, Unmute's STT model outputs a continuous **pause prediction score** (0.0 to 1.0):

```python
# unmute_handler.py — determine_pause()
def determine_pause(self) -> bool:
    if self.chatbot.conversation_state() != "user_speaking":
        return False
    if stt.pause_prediction.value > 0.6:  # Semantic threshold
        return True
    return False
```

**What the score considers:**
- Linguistic completeness (is the sentence finished?)
- Intonation patterns (rising pitch = still talking)
- Syntactic structure (trailing conjunction = not done)

**Threshold: 0.6** — Only triggers when 60%+ confident the user finished their thought.

### 4.2 Response Pipeline Sequence

1. **Pause detected** → `determine_pause()` returns True
2. **Flush STT buffer** — Send zero-padded audio frames to get final transcription
3. **Generate response** — Create asyncio task for `_generate_response_task()`
4. **Start TTS** — Initialize TTS WebSocket connection
5. **Stream LLM → TTS** — Each word from LLM is sent to TTS immediately via `rechunk_to_words()`
6. **Stream TTS → Browser** — TTS audio frames pushed to output queue
7. **Finish** — Signal end-of-turn, add empty user message to transition state

### 4.3 Word-Level Response Chunking

Unlike sentence-level chunking, Unmute sends **each word** to TTS immediately:

```python
async def rechunk_to_words(iterator):
    buffer = ""
    space_re = re.compile(r"\s+")
    prefix = ""
    async for delta in iterator:
        buffer = buffer + delta
        while True:
            match = space_re.search(buffer)
            if match is None:
                break
            chunk = buffer[:match.start()]
            buffer = buffer[match.end():]
            if chunk != "":
                yield prefix + chunk
            prefix = " "
    if buffer != "":
        yield prefix + buffer
```

This works because Kyutai's TTS is a **streaming model** with lookahead — it can start speaking before seeing the full sentence.

### 4.4 Interruption Handling

Two mechanisms:

**A. STT-based (text detection):**
```python
if self.chatbot.conversation_state() == "bot_speaking":
    logger.info("STT-based interruption")
    await self.interrupt_bot()
```

**B. VAD-based (audio detection):**
```python
if (self.chatbot.conversation_state() == "bot_speaking"
    and stt.pause_prediction.value < 0.4
    and self.audio_received_sec() > UNINTERRUPTIBLE_BY_VAD_TIME_SEC):
    await self.interrupt_bot()
```

**`interrupt_bot()` sequence:**
1. Mark assistant message with `—` (em-dash)
2. Clear output queue (discard buffered audio)
3. Replace queue with new empty one
4. Push silence frame to flush Opus codec
5. Cancel TTS and LLM tasks

### 4.5 Long Silence Recovery

```python
USER_SILENCE_TIMEOUT = 7.0

async def detect_long_silence(self):
    if (self.chatbot.conversation_state() == "waiting_for_user"
        and (self.audio_received_sec() - self.waiting_for_user_start_time) > USER_SILENCE_TIMEOUT):
        await self.add_chat_message_delta(USER_SILENCE_MARKER, "user")  # "..."
```

The system prompt instructs the LLM how to handle `"..."` — typically with encouraging nudges.

### 4.6 Chat History Preprocessing

Before every LLM call:
```python
def preprocess_messages_for_llm(chat_history):
    # 1. Remove empty/interruption-only messages
    # 2. Strip trailing em-dash interruption markers
    # 3. Merge consecutive same-role messages
    # 4. Add dummy user message if needed for models like Gemma
    # 5. Clean silence markers from messages where user resumed talking
```

---

## 5. Gap Analysis: Fast_tutor vs Unmute

| Feature | Fast_tutor (Before) | Unmute | Fast_tutor (After) |
|---------|-------------------|--------|-------------------|
| **VAD** | RMS amplitude + fixed 400ms | Semantic pause prediction (0.0-1.0) | Heuristic transcript completeness (0.0-1.0) + adaptive silence (0.3s-1.2s) |
| **State management** | `is_tts_playing` boolean | Derived from chat history | `ConversationFSM` class with 3 states |
| **Response chunking** | Sentence-level (`.?!,`) | Word-level (`rechunk_to_words`) | Sentence-level with regex (kept due to batch TTS) |
| **TTS** | Pocket-TTS (HTTP batch) | Kyutai TTS (WebSocket streaming) | Pocket-TTS (HTTP batch) — unchanged |
| **Interruption** | Mic fully muted during playback | Dual: STT text + VAD score, 3s grace | Volume threshold + 3s grace period |
| **Self-echo prevention** | `is_tts_playing` flag | 3s uninterruptible + Opus echo cancellation | 3s grace + mic buffer clearing |
| **Long silence** | Not handled | 7s → "..." marker | 7s first nudge, 20s repeat |
| **History cleanup** | Simple rolling window (6) | Merge fragments, strip markers | Merge fragments, strip markers (8 window) |
| **Name tracking** | Not handled | N/A (not a tutor) | Auto-extract from first response, use throughout |
| **Math-only focus** | Generic tutor prompt | N/A | Strict math-only rules in system prompt |

---

## 6. Implementation Plan

### Constraints
- **STT**: Moonshine ONNX (`moonshine/tiny`) — no semantic VAD
- **LLM**: Qwen 2.5:0.5b via Ollama — small model, limited instruction following
- **TTS**: Pocket-TTS — HTTP batch mode, not streaming
- **All changes**: Single file (`orchestrator.py`), no new dependencies

### 7 Components Planned

1. **ConversationFSM** — Replace `is_tts_playing` with proper state machine
2. **Heuristic Semantic Pause Detection** — Analyze transcript text for completeness
3. **Smart Interruption** — 3s grace period, higher volume threshold
4. **Long Silence Recovery** — 7s timeout → "..." nudge
5. **Chat History Preprocessing** — Merge, clean, strip markers
6. **Smarter TTS Chunking** — Regex sentence splitting
7. **Post-Playback Pause** — 500ms breathing room

### Design Decision: Why Not Word-Level TTS?

Pocket-TTS is an HTTP batch endpoint (`POST /tts`). It needs the full text before generating audio. Word-level streaming requires a WebSocket-based streaming TTS (like Kyutai's). We kept sentence-level chunking as the optimal approach for our TTS.

---

## 7. What Was Implemented

### Component 1: ConversationFSM

**Lines 80-131** in `orchestrator.py`

```python
class ConversationFSM:
    """Centralized 3-state FSM."""
    def __init__(self):
        self.state = "waiting_for_user"
        self.bot_speaking_start_time = 0.0
        self.waiting_start_time = time.time()
        self.silence_nudge_count = 0
        self.current_llm_task: asyncio.Task | None = None
```

States: `waiting_for_user` → `user_speaking` → `bot_speaking` → `waiting_for_user`

Every transition is logged: `[STATE] waiting_for_user → user_speaking`

### Component 2: Heuristic Semantic Pause Detection

**Lines 172-220** — Two functions:

```python
def estimate_transcript_completeness(transcript: str) -> float:
    """Score 0.0-1.0 based on linguistic heuristics."""
    # Checks: punctuation, trailing words, word count, numeric endings
    
def get_adaptive_silence_duration(running_transcript: str) -> float:
    """Maps completeness to silence threshold (0.3s-1.2s)."""
```

| Input | Completeness | Silence Duration |
|-------|-------------|-----------------|
| "The answer is four." | 0.9 | 0.3s (fast response) |
| "I think the answer is" | 0.2 | 1.1s (patient wait) |
| "Um" | 0.2 | 1.1s (patient wait) |
| "42" | 0.7 | 0.5s (likely answer) |

### Component 3: Interruption with Grace Period

**Lines 224-340** — Three changes:

1. `audio_callback()` — checks `INTERRUPT_THRESHOLD = 0.06` and `UNINTERRUPTIBLE_GRACE_SEC = 3.0`
2. `_interrupt_flag` — thread-safe flag from C callback to async loop
3. `trigger_interrupt()` — cancels LLM, clears queues, marks history with `—`

### Component 4: Long Silence Recovery

**Lines 637-648** in `audio_loop()`:

```python
USER_SILENCE_TIMEOUT = 7.0          # First nudge
USER_SILENCE_REPEAT_TIMEOUT = 20.0  # Subsequent nudges

if conv_state.is_waiting and conv_state.waiting_elapsed > timeout:
    process_llm_and_speak(USER_SILENCE_MARKER, global_page_state)
```

### Component 5: Chat History Preprocessing

**Lines 144-166**:

```python
def preprocess_history(history):
    # 1. Skip empty / "—" only messages
    # 2. Strip trailing "—"
    # 3. Merge consecutive same-role messages
```

### Component 6: Smarter TTS Chunking

**Lines 540-545** in `process_llm_and_speak()`:

```python
if re.search(r'[.!?]\s*$', sentence_buffer) or "\n" in token:
    sentence_to_speak = sentence_buffer.strip()
    if len(sentence_to_speak) > 2:
        await speak(sentence_to_speak)
```

### Component 7: Post-Playback Pause

**Lines 401-414** in `audio_playback_worker()`:

```python
if audio_playback_queue.empty() and tts_text_queue.empty():
    await asyncio.sleep(POST_PLAYBACK_PAUSE)  # 500ms
    conv_state.transition("waiting_for_user")
```

### Name Tracking & Math-Only Context (Added Later)

**`extract_name()`** — Regex-based name extraction from first user response:
- Handles: "My name is Ali", "I'm Sara", "Ali", "It's Mohamed"
- Excludes common non-name words
- Logged: `[NAME] Child's name detected: Ali`

**`get_system_prompt()`** — Dynamic system prompt:
- Before name: "Ask for name first"
- After name: "Child's name is {name}, use it naturally"
- Math-only rules: redirects non-math topics
- Silence handling: personalized nudges with name

---

## 8. Testing Results

### Live Pipeline Test (April 22, 2026)

```
[1/3] Booting Voice Engine (Pocket-TTS)...
[2/3] Checking LLM Engine (Ollama)... ✅ active
[3/3] Booting Python Backbone...

Pipeline Ready! Start talking into your default microphone.
```

**Verified behaviors:**

| Test Case | Result | Log Evidence |
|-----------|--------|-------------|
| State transitions | ✅ | `[STATE] waiting_for_user → user_speaking → bot_speaking → waiting_for_user` |
| Greeting plays | ✅ | TTS fetches and plays greeting after 4s delay |
| STT → LLM → TTS | ✅ | User speech transcribed, LLM responds, sentence-chunked to TTS |
| Sentence chunking | ✅ | Response split into natural chunks: "Got it!", "Welcome to..." |
| Post-playback pause | ✅ | Clean `bot_speaking → waiting_for_user` after 500ms pause |
| 7s silence nudge | ✅ | `[SILENCE] User silent for 7.0s, nudging...` fires correctly |
| Noise rejection | ✅ | Static noise correctly ignored as `<Nothing coherent transcribed>` |
| Repeat nudge | ✅ | Second nudge fires after subsequent silence |

### Configuration Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `SILENCE_THRESHOLD` | 0.02 | RMS threshold for speech detection |
| `INTERRUPT_THRESHOLD` | 0.06 | Higher threshold during bot speech |
| `MIN_SILENCE_DURATION` | 0.3s | Fast response for complete sentences |
| `MAX_SILENCE_DURATION` | 1.2s | Patient wait for incomplete thoughts |
| `UNINTERRUPTIBLE_GRACE_SEC` | 3.0s | Anti-echo grace period |
| `USER_SILENCE_TIMEOUT` | 7.0s | First silence nudge |
| `USER_SILENCE_REPEAT_TIMEOUT` | 20.0s | Repeat nudge interval |
| `POST_PLAYBACK_PAUSE` | 0.5s | Breathing room after bot speaks |

---

## References

- [Kyutai Labs — Unmute](https://github.com/kyutai-labs/unmute)
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling)
- [Unmute Research Paper](https://arxiv.org/pdf/2509.08753)
- [Kyutai Blog](https://kyutai.org/)
- [Moonshine ONNX STT](https://huggingface.co/UsefulSensors/moonshine)
- [Pocket-TTS](https://pypi.org/project/pocket-tts/)
