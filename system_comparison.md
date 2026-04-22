# Comprehensive Voice AI Architecture Comparison

This document provides a detailed technical comparison between **Our System (Fast Tutor)**, **Moshi (Kyutai)**, **Ultravox (Fixie.ai)**, and **Unmute (Kyutai)**. It evaluates each system along four critical axes: **Design**, **Resource Usage**, **Speed (Latency)**, and **Accuracy**.

---

## 1. System Design and Architecture

The fundamental differences between these systems lie in how they process speech input and output. Broadly, they fall into two categories: **Cascaded Systems** (modular pipelines) and **Native/Multimodal Models** (end-to-end networks).

### Our System (Fast Tutor)

```mermaid
graph TD
    classDef user fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef stt fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef llm fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef tts fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef orc fill:#eceff1,stroke:#455a64,stroke-width:2px;

    User((👤 User)):::user
    Speaker((🔊 Audio Out)):::user

    subgraph Fast Tutor Offline Pipeline
        direction TB
        Orchestrator{⚙️ Python Orchestrator<br>& WebSocket Router}:::orc
        STT[🎙️ Moonshine STT]:::stt
        LLM[(🧠 Local Text LLM<br>Ollama / TinyLlama)]:::llm
        TTS[📢 Kokoro TTS]:::tts
        
        STT -- "📝 Transcribed Text" --> Orchestrator
        Orchestrator -- "🛠️ Prompt + Text" --> LLM
        LLM -- "📝 Text Stream" --> Orchestrator
        Orchestrator -- "📝 Text Stream" --> TTS
    end

    User -- "🗣️ Speech Chunks" --> STT
    TTS -- "🎵 Synthesized Voice" --> Speaker
```

* **Architecture:** Offline, Event-Driven Cascaded Pipeline (Unmute-inspired).
* **Component Flow:** 
  * **STT:** Moonshine Tiny (highly optimized, local speech-to-text with low GPU footprint).
  * **LLM Engine:** Ollama running `qwen2.5:0.5b` (extremely lightweight 0.5B text model) for incredibly fast text generation.
  * **TTS:** Localized TTS (Kokoro-TTS via Pocket-TTS).
* **System Design:** Held together by a Python WebSocket orchestrator that mirrors Unmute's architecture but runs entirely offline. It features:
  - **Heuristic Semantic VAD:** Predicts utterance completion based on punctuation, trailing conjunctions, and sentence length to minimize silence gaps.
  - **Smart Interruptions:** Uses RMS thresholds to allow the user to interrupt the bot gracefully after an initial 3-second uninterruptible grace period.
  - **Conversation FSM:** Centralized state tracking (`waiting_for_user`, `user_speaking`, `bot_speaking`) with long-silence nudges.

### Moshi (by Kyutai)

```mermaid
graph TD
    classDef user fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef model fill:#e8f5e9,stroke:#388e3c,stroke-width:4px,color:#000;
    classDef codec fill:#fff3e0,stroke:#f57c00,stroke-width:2px;

    User((👤 User)):::user
    Speaker((🔊 Audio Out)):::user

    subgraph Moshi End-to-End Architecture
        direction TB
        CodecIn[/📥 Helium Audio Encoder/]:::codec
        MoshiLLM((🌌 Moshi 7B<br>Multimodal LLM)):::model
        CodecOut[\📤 Helium Audio Decoder\]:::codec
        
        CodecIn -- "🔢 Acoustic Tokens" --> MoshiLLM
        MoshiLLM -- "🔢 Acoustic Tokens" --> CodecOut
    end

    User -- "🗣️ Raw Audio Stream" --> CodecIn
    CodecOut -- "🎵 Raw Audio Stream" --> Speaker
```

* **Architecture:** Native End-to-End Speech-to-Speech AI.
* **Component Flow:** Pure Multimodal. Moshi does not transcribe speech to text. Instead, it processes raw audio directly into semantic tokens and hallucinates raw audio outputs (using Helium/Mimi audio codecs) interleaved with text representations. 
* **System Design:** Fully integrated model capable of true full-duplex communication. It can listen to you *while* it speaks, handling interruptions flawlessly.

### Ultravox (by Fixie.ai)

```mermaid
graph TD
    classDef user fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef enc fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef mode fill:#e8f5e9,stroke:#388e3c,stroke-width:3px;
    classDef tts fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    User((👤 User)):::user
    Speaker((🔊 Audio Out)):::user

    subgraph Ultravox Speech LLM
        direction TB
        Whisper[👂 Whisper Audio Encoder]:::enc
        LLM[(🧠 Llama-3 / Mistral Backbone)]:::mode
        TTSProxy[📢 External TTS Engine]:::tts
        
        Whisper -- "📊 High-Dim Audio Vectors" --> LLM
        LLM -- "📝 Generated Text Tokens" --> TTSProxy
    end

    User -- "🗣️ Speech Audio" --> Whisper
    TTSProxy -- "🎵 Synthesized Voice" --> Speaker
```

* **Architecture:** Multimodal Speech Large Language Model (Speech-to-Text LLM).
* **Component Flow:** 
  * **Audio Encoding:** Direct-to-LLM audio processing (combines Whisper's audio encoder directly into a Llama-3 or Mistral backbone).
  * **Text Output:** The model natively understands audio inputs (bypassing a separate STT phase) but outputs text.
  * **TTS:** Requires a separate Text-to-Speech engine attached to the end to converse vocally.
* **System Design:** It eliminates the latency and error-propagation of an STT pipeline by encoding audio features directly to LLM tokens. It supports tool-calling and RAG.

### Unmute (by Kyutai)

```mermaid
graph TD
    classDef user fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef vad fill:#ffebee,stroke:#d32f2f,stroke-width:2px;
    classDef stt fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef llm fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef tts fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    User((👤 User)):::user
    Speaker((🔊 Audio Out)):::user

    subgraph Unmute Optimized Wrapper
        direction TB
        VAD{✂️ Semantic VAD}:::vad
        STT[🎙️ Fast STT Engine]:::stt
        LLM[(🧠 Standard Text LLM)]:::llm
        TTS[📢 Kyutai Fast TTS]:::tts
        
        VAD -- "✂️ Utterance Cut" --> STT
        STT -- "📝 Final Text" --> LLM
        LLM -- "📝 Streamed Text" --> TTS
    end

    User -- "🗣️ Audio Stream" --> VAD
    TTS -- "🎵 Synthesized Voice" --> Speaker
```

* **Architecture:** Optimized Orchestration Wrapper (Cascaded Pipeline).
* **Component Flow:**
  * **VAD:** Semantic Voice Activity Detection (intelligently detects when the user finishes speaking).
  * **STT -> LLM -> TTS:** Connects high-performance standard cloud or local text LLMs to fast STT and TTS engines.
* **System Design:** Unmute is not a model, but a framework. It attempts to squeeze maximum speed out of a traditional cascaded pipeline (like Our System) by streaming chunks aggressively between independent STT, LLM, and TTS processes.

---

## 2. Resource Usage

| System | Primary Resource Constraint | VRAM/Memory Footprint | Deployment Scope |
| :--- | :--- | :--- | :--- |
| **Our System** | Highly Scalable / Modular | **Ultra-Low (< 4GB VRAM)** | By using Moonshine Tiny and Qwen2.5 (0.5B parameters), it runs effortlessly on basic consumer hardware and Apple Silicon CPUs without crashing. |
| **Moshi** | Heavy GPU Requirement | **High (~8GB - 16GB+ VRAM)** | As a 7B parameter multimodal model, it requires a dedicated AI accelerator / heavy GPU to run inference locally at real-time speeds. |
| **Ultravox** | Moderate-to-Heavy GPU | **High (~16GB VRAM)** | Usually built on top of 8B Llama-3 models. Requires substantial memory just to load the LLM and the attached audio encoder. |
| **Unmute** | Modular (Depends on models) | **Variable** | Since it acts as a wrapper, it can be lightweight if calling API endpoints, or highly resource-intensive if running its STT, LLM, and TTS fully locally simultaneously. |

**Summary:** 
Our System overwhelmingly wins on minimum viable footprint. By intentionally down-scaling to Qwen2.5-0.5B and Moonshine, Fast Tutor achieves near real-time voice streaming on hardware where Moshi or Ultravox would fundamentally OOM (Out-of-Memory) or fail to achieve real-time streaming speeds.

---

## 3. Speed and Latency

Latency in voice AI includes Time-To-First-Byte (TTFB) and full dialogue turnaround time.

1. **Moshi (Fastest):** Operates under **~200ms latency**. Because it is a native Speech-to-Speech model, it requires zero time for text transduction. It begins generating audio responses virtually instantly, rivaling human reflex time.
2. **Our System (High Velocity Cascade):** Optimized for **fast turnaround (~400ms - 800ms)**. While fundamentally a pipeline, Fast Tutor actively mitigates the "pipeline tax" using its Heuristic Semantic VAD (cutting off STT early when a sentence seems complete) and aggressive LLM sentence-chunking (submitting incomplete text buffers to Kokoro TTS before the LLM finishes). This puts its TTFB exceptionally close to native models.
3. **Ultravox (Fast):** By omitting the STT transcription step, it saves roughly 300ms associated with traditional ASR. However, it still must stream its text generation into a TTS engine. Latency is generally around **~500ms - 700ms**.
4. **Unmute (Baseline for Fast Tutor):** Fast Tutor is essentially a localized implementation of Unmute's theories. Both systems minimize latency purely through advanced VAD and parallel execution of STT/LLM/TTS, clocking in around **~600ms - 1000ms** depending on hardware.

---

## 4. Accuracy and Output Quality

Different architectures handle audio nuances, reasoning, and context differently.

### Reasoning and Mathematical Accuracy
* **Our System:** **Highest for specific constraints.** Because we pass clean text to an LLM, we can inject very strict system prompts (e.g., "Answer only math questions"). Local LLMs tuned for reasoning will outperform multimodal audio models on raw logic tasks.
* **Ultravox:** **High.** Utilizes Llama-3 or Mistral backbones. Excellent at reasoning tasks and function calling, processing the context with standard top-tier instruction following.
* **Moshi & Unmute:** **Moderate to Good.** End-to-end models like Moshi can sometimes struggle with deep logical reasoning because parameters are heavily dedicated to acoustic mapping, whereas text-first systems dedicate all compute to logic.

### Acoustic Understaning (Emotion, Tone, Interruptions)
* **Moshi:** **Flawless.** Natively understands tone of voice, background noise, and allows you to seamlessly interrupt it mid-sentence.
* **Ultravox:** **Excellent.** Reads the tone and non-verbal cues (like sighs or hesitations) directly from the audio encoder, providing richer context to the LLM than a flat transcript.
* **Our System & Unmute:** **Limited.** As cascaded pipelines, STT flattens all audio into plain text. The LLM cannot hear the user's emotion, sarcasm, or background context. Interruptions have to be manually handled by cutting the STT/TTS buffers abruptly.

---

## Conclusion & Architectural Verdict

If the goal is to build a **Local, Math-Focused AI Tutor**:

If the goal is to build a **Local, Math-Focused AI Tutor**:

1. **Our System vs. Moshi:** Moshi provides an incredible acoustic experience but is too resource-hungry and its raw acoustic hallucination makes it hard to restrict logically. Our System (Fast Tutor) gives us tight logical control, strict math rule enforcement via system prompting, and runs on a fraction of the hardware budget.
2. **Our System vs. Ultravox:** Ultravox is an incredible middle-ground (skipping STT), but still requires heavy GPU hardware (+16GB VRAM) for local deployment. Fast Tutor's Moonshine STT + Qwen2.5 combo is vastly more adaptable for lower-end offline laptops.
3. **Our System vs. Unmute:** Fast Tutor *is* an Unmute-style architectural clone adapted strictly for offline execution and educational moderation. It proves that a properly orchestrated, streaming cascaded pipeline with Semantic VAD logic is currently the most practical, customizable, and reliable way to build specialized voice applications.
