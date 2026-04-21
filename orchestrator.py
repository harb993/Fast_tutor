import asyncio
import httpx
import json
import logging
import queue
import re
import time
import io
import os

import numpy as np
import sounddevice as sd
import soundfile as sf
from moonshine_onnx import MoonshineOnnxModel, load_tokenizer

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("orchestrator")

# Ports Configuration
TTS_PORT = os.environ.get("TTS_PORT", "8000")
OLLAMA_PORT = os.environ.get("OLLAMA_PORT", "11434")
ORCHESTRATOR_PORT = int(os.environ.get("ORCHESTRATOR_PORT", "8001"))

# Pipeline Endpoints
OLLAMA_SERVER = f"http://localhost:{OLLAMA_PORT}/v1/chat/completions"
TTS_SERVER = f"http://localhost:{TTS_PORT}/tts"

# ============================================================
# Audio & Conversation Constants (Unmute-inspired)
# ============================================================
SAMPLE_RATE = 16000
CHANNELS = 1

# VAD Thresholds
SILENCE_THRESHOLD = 0.02           # RMS threshold to detect speech
INTERRUPT_THRESHOLD = 0.06         # Higher threshold for interruption during bot speech

# Adaptive Silence Duration (Component 2)
MIN_SILENCE_DURATION = 0.3         # Fast response for complete sentences
MAX_SILENCE_DURATION = 1.2         # Patient wait for incomplete thoughts
DEFAULT_SILENCE_DURATION = 0.5     # Fallback

# Interruption (Component 3)
UNINTERRUPTIBLE_GRACE_SEC = 3.0    # Don't allow interruption for first 3s of bot speech

# Long Silence Recovery (Component 4)
USER_SILENCE_TIMEOUT = 7.0         # Seconds before nudging a silent user
USER_SILENCE_REPEAT_TIMEOUT = 20.0 # After first nudge, wait longer before re-nudging

# Conversation markers (Component 5)
INTERRUPTION_MARKER = "—"          # Em-dash marks an interrupted response
USER_SILENCE_MARKER = "..."        # Inserted when user is silent too long

# Post-playback pause (Component 7)
POST_PLAYBACK_PAUSE = 0.5         # 500ms breathing room after bot finishes speaking

# ============================================================
# Application Setup
# ============================================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

connected_clients = []
global_page_state = {"lesson": "Starting up", "status": "Ready"}

# ============================================================
# Component 1: Conversation State Machine (Unmute-inspired)
# ============================================================
class ConversationFSM:
    """
    Centralized 3-state FSM replacing the scattered is_tts_playing flag.
    States: waiting_for_user | user_speaking | bot_speaking
    """
    def __init__(self):
        self.state = "waiting_for_user"
        self.bot_speaking_start_time = 0.0
        self.waiting_start_time = time.time()
        self.silence_nudge_count = 0
        self.current_llm_task: asyncio.Task | None = None

    def transition(self, new_state: str):
        if new_state == self.state:
            return
        logger.info(f"[STATE] {self.state} → {new_state}")
        old_state = self.state
        self.state = new_state

        if new_state == "bot_speaking":
            self.bot_speaking_start_time = time.time()
        elif new_state == "waiting_for_user":
            self.waiting_start_time = time.time()
            self.silence_nudge_count = 0
        elif new_state == "user_speaking":
            pass  # Reset handled by VAD loop

    @property
    def is_bot_speaking(self) -> bool:
        return self.state == "bot_speaking"

    @property
    def is_waiting(self) -> bool:
        return self.state == "waiting_for_user"

    @property
    def is_user_speaking(self) -> bool:
        return self.state == "user_speaking"

    @property
    def bot_speaking_elapsed(self) -> float:
        """Seconds since bot started speaking."""
        if self.state != "bot_speaking":
            return 0.0
        return time.time() - self.bot_speaking_start_time

    @property
    def waiting_elapsed(self) -> float:
        """Seconds since we started waiting for user."""
        if self.state != "waiting_for_user":
            return 0.0
        return time.time() - self.waiting_start_time


conv_state = ConversationFSM()

# Child's name — extracted from first interaction
child_name: str | None = None

# Chat history — kept as a rolling window
chat_history: list[dict] = [
    {"role": "assistant", "content": "Hello! I am your Math Tutor. What is your name?"}
]

# ============================================================
# Component 5: Chat History Preprocessing (Unmute-style)
# ============================================================
def preprocess_history(history: list[dict]) -> list[dict]:
    """
    Clean chat history before sending to LLM.
    - Removes empty / interruption-only messages
    - Strips trailing interruption markers
    - Merges consecutive same-role messages into one
    """
    output = []
    for msg in history:
        content = msg["content"].strip()
        # Skip empty or interruption-only entries
        if not content or content == INTERRUPTION_MARKER:
            continue
        # Strip trailing interruption markers
        content = content.rstrip(INTERRUPTION_MARKER).strip()
        if not content:
            continue
        # Merge consecutive same-role messages
        if output and msg["role"] == output[-1]["role"]:
            output[-1]["content"] += " " + content
        else:
            output.append({"role": msg["role"], "content": content})
    return output


# ============================================================
# Component 2: Heuristic Semantic Pause Detection
# ============================================================
def estimate_transcript_completeness(transcript: str) -> float:
    """
    Heuristic 0.0-1.0 score estimating if the user finished their thought.
    Approximates Unmute's Semantic VAD without a dedicated model.
    """
    text = transcript.strip()
    if not text:
        return 0.0

    score = 0.5  # neutral baseline

    # Strong end-of-turn signals (punctuation)
    if text[-1] in '.!?':
        score += 0.3

    last_word = text.split()[-1].lower().rstrip('.,!?')

    # Trailing conjunctions / prepositions / operators = clearly incomplete
    incomplete_endings = [
        'and', 'but', 'or', 'the', 'a', 'an', 'to', 'is', 'are', 'was',
        'plus', 'minus', 'times', 'divided', 'equals', 'if', 'then',
        'because', 'so', 'like', 'um', 'uh',
    ]
    if last_word in incomplete_endings:
        score -= 0.3

    # Very short utterances are likely incomplete
    word_count = len(text.split())
    if word_count <= 2:
        score -= 0.1

    # Numbers at the end (child answering a math problem) = likely complete
    if last_word.isdigit():
        score += 0.2

    # Longer complete sentences are more likely finished
    if word_count >= 5 and text[-1] in '.!?':
        score += 0.1

    return max(0.0, min(1.0, score))


def get_adaptive_silence_duration(running_transcript: str) -> float:
    """
    Adaptive silence threshold: less patience for complete thoughts,
    more patience for incomplete ones.
    """
    if not running_transcript.strip():
        return DEFAULT_SILENCE_DURATION

    completeness = estimate_transcript_completeness(running_transcript)
    # High completeness → short wait, low completeness → long wait
    duration = MAX_SILENCE_DURATION - (completeness * (MAX_SILENCE_DURATION - MIN_SILENCE_DURATION))
    return duration


# ============================================================
# Component 6: Smarter TTS Sentence Chunking
# ============================================================
def split_into_speakable_chunks(text: str) -> list[str]:
    """
    Split a buffer into natural speech chunks on sentence boundaries.
    Returns (chunks_to_speak, remaining_buffer).
    """
    # Split on sentence-ending punctuation followed by space (or end of string)
    chunks = re.split(r'(?<=[.!?])\s+', text)
    return [c.strip() for c in chunks if len(c.strip()) > 2]


# ============================================================
# WebSocket & Broadcasting
# ============================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get('type') == 'page_state':
                    global global_page_state
                    global_page_state = msg.get('state')
                elif msg.get('type') == 'user_chat':
                    user_text = msg.get('text', '')
                    sys_prompt = msg.get('system_prompt', None)
                    if user_text:
                        asyncio.create_task(process_llm_and_speak(user_text, global_page_state, sys_prompt))
            except Exception:
                pass
    except WebSocketDisconnect:
        connected_clients.remove(websocket)


async def broadcast(message: dict):
    for client in connected_clients:
        try:
            await client.send_text(json.dumps(message))
        except Exception:
            pass


# ============================================================
# STT Model Initialization
# ============================================================
logger.info("Initializing Moonshine STT model... (this may take a moment)")
stt = MoonshineOnnxModel(model_name="moonshine/tiny")
tokenizer = load_tokenizer()
logger.info("Moonshine STT model loaded successfully.")


# ============================================================
# Audio Pipeline
# ============================================================
audio_queue = queue.Queue()
tts_text_queue = asyncio.Queue()
audio_playback_queue = asyncio.Queue()
_interrupt_flag = False  # Set from audio callback thread, consumed by async loop


def audio_callback(indata, frames, time_info, status):
    """
    Component 3: Smarter mic handling during bot speech.
    Instead of fully muting the mic, we use a higher threshold + grace period.
    """
    global _interrupt_flag
    if status:
        logger.warning(f"Audio status: {status}")

    if conv_state.is_bot_speaking:
        rms = np.sqrt(np.mean(indata**2))
        # Only allow interruption after the grace period
        if (conv_state.bot_speaking_elapsed > UNINTERRUPTIBLE_GRACE_SEC
                and rms > INTERRUPT_THRESHOLD):
            audio_queue.put(indata.copy())
            _interrupt_flag = True
        # Otherwise: ignore mic input to prevent echo feedback
    else:
        audio_queue.put(indata.copy())


# ============================================================
# Component 3: Interrupt Handler
# ============================================================
async def trigger_interrupt():
    """Cancel ongoing LLM/TTS and return to listening."""
    global chat_history

    if not conv_state.is_bot_speaking:
        return

    logger.info("[INTERRUPT] User interrupted bot! Stopping generation...")

    # Mark the assistant's response as interrupted
    if chat_history and chat_history[-1]["role"] == "assistant":
        chat_history[-1]["content"] += INTERRUPTION_MARKER

    # Cancel LLM task if running
    if conv_state.current_llm_task and not conv_state.current_llm_task.done():
        conv_state.current_llm_task.cancel()
        conv_state.current_llm_task = None

    # Clear TTS text queue
    while not tts_text_queue.empty():
        try:
            tts_text_queue.get_nowait()
            tts_text_queue.task_done()
        except asyncio.QueueEmpty:
            break

    # Clear audio playback queue
    while not audio_playback_queue.empty():
        try:
            audio_playback_queue.get_nowait()
            audio_playback_queue.task_done()
        except asyncio.QueueEmpty:
            break

    # Stop any active playback
    sd.stop()

    # Clear mic buffer to remove echo
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break

    conv_state.transition("waiting_for_user")
    await broadcast({"type": "tutor_interrupted"})


# ============================================================
# Playback & TTS Workers
# ============================================================
async def audio_playback_worker():
    """
    Component 7: Post-playback breathing room.
    After the last chunk plays, wait 500ms before transitioning to waiting_for_user.
    """
    while True:
        data, samplerate = await audio_playback_queue.get()

        # If we've been interrupted, discard this chunk
        if not conv_state.is_bot_speaking:
            audio_playback_queue.task_done()
            continue

        # Clear mic buffer before playing to prevent echo
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break

        def _play():
            sd.play(data, samplerate)
            sd.wait()

        await asyncio.to_thread(_play)

        # After playing, clear any echo that leaked into mic
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break

        # Check if this was the last chunk
        if (audio_playback_queue.empty()
                and tts_text_queue.empty()
                and (conv_state.current_llm_task is None or conv_state.current_llm_task.done())):
            if conv_state.is_bot_speaking:
                # Component 7: breathing room before listening
                await asyncio.sleep(POST_PLAYBACK_PAUSE)
                # Clear any noise captured during pause
                while not audio_queue.empty():
                    try:
                        audio_queue.get_nowait()
                    except queue.Empty:
                        break
                conv_state.transition("waiting_for_user")

        audio_playback_queue.task_done()


async def tts_fetch_worker():
    """Fetches audio from Pocket-TTS for each text chunk."""
    while True:
        text = await tts_text_queue.get()

        # Discard if interrupted
        if not conv_state.is_bot_speaking:
            tts_text_queue.task_done()
            continue

        logger.info(f"[TTS Worker] Fetching audio for: '{text}'")
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(TTS_SERVER, data={"text": text}, timeout=15.0)
                if r.status_code == 200:
                    audio_bytes = r.content
                    with io.BytesIO(audio_bytes) as f:
                        data, samplerate = sf.read(f)
                    # Only enqueue if still speaking (might have been interrupted during TTS fetch)
                    if conv_state.is_bot_speaking:
                        await audio_playback_queue.put((data, samplerate))
                else:
                    logger.error(f"[TTS] Error from TTS server: {r.status_code}")
        except Exception as e:
            logger.error(f"[TTS Worker] Communication exception: {e}")
        tts_text_queue.task_done()


async def speak(text: str):
    """Queues text to be processed by the TTS worker."""
    if conv_state.is_bot_speaking:
        await tts_text_queue.put(text)


# ============================================================
# Prompt Building (Component 5 integration + Name Tracking)
# ============================================================

def extract_name(text: str) -> str | None:
    """
    Try to extract a child's name from their first response.
    Handles patterns like: 'My name is Ali', 'I'm Sara', 'Ali', 'It's Mohamed', etc.
    Conservative: requires 3+ letter names and filters aggressively to avoid STT noise.
    """
    text = text.strip().rstrip('.!?,')
    import re as _re

    # Words that Moonshine STT commonly hallucinates or that aren't names
    NOT_NAMES = {
        'the', 'yes', 'no', 'and', 'okay', 'math', 'hello', 'hi', 'hey',
        'what', 'how', 'why', 'think', 'right', 'good', 'fine', 'well',
        'name', 'is', 'am', 'are', 'was', 'his', 'her', 'its', 'this',
        'that', 'here', 'there', 'have', 'has', 'had', 'can', 'could',
        'will', 'would', 'should', 'did', 'does', 'done', 'just', 'like',
        'know', 'think', 'want', 'need', 'let', 'put', 'get', 'got',
        'see', 'say', 'said', 'come', 'going', 'take', 'make', 'made',
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
        'nine', 'ten', 'lot', 'much', 'many', 'some', 'all', 'not',
        'but', 'for', 'with', 'from', 'about', 'into', 'over', 'also',
        'mics', 'mic', 'mike', 'plus', 'minus', 'times', 'answer',
        'number', 'problem', 'lesson', 'tutor', 'start', 'ready',
    }

    def is_valid_name(candidate: str) -> bool:
        """A valid name is 3+ alphabetic chars and not a common English word."""
        c = candidate.strip().capitalize()
        return (len(c) >= 3
                and c.isalpha()
                and c.lower() not in NOT_NAMES)

    # Priority 1: Explicit "my name is X" pattern (most reliable)
    m = _re.search(r"(?:my name is|i'm|i am|it's|call me|they call me)\s+([a-zA-Z]{3,})", text, _re.IGNORECASE)
    if m:
        name = m.group(1).strip().capitalize()
        if is_valid_name(name):
            return name

    # Priority 2: Single capitalized word (like "Ali" or "Sara")
    m = _re.match(r"^([A-Z][a-z]{2,15})$", text.strip())
    if m:
        name = m.group(1).capitalize()
        if is_valid_name(name):
            return name

    # Priority 3: Short response (1-2 words) where last word looks like a name
    words = text.split()
    if 1 <= len(words) <= 2:
        candidate = words[-1].strip().capitalize()
        if is_valid_name(candidate):
            return candidate

    return None


def get_system_prompt() -> str:
    """Build the system prompt dynamically based on whether we know the child's name."""
    name_line = ""
    if child_name:
        name_line = f"The child's name is {child_name}. Use their name naturally in your responses to make them feel special.\n"
    else:
        name_line = "You don't know the child's name yet. Your very first priority is to warmly ask for their name before doing any math.\n"

    return f"""You are a friendly, encouraging math tutor for a child. You ONLY help with math.
{name_line}
CORE RULES:
1. ONLY answer math-related questions. If the child asks about anything else (stories, games, animals, etc.), gently redirect: "{child_name + ', t' if child_name else 'T'}hat sounds fun, but let's focus on math! What's the answer to this problem?"
2. Keep responses to 1-2 sentences maximum. Be concise.
3. Speak in clear, simple, natural language appropriate for a child.
4. Guide the child step-by-step. Don't give away the answer — help them figure it out.
5. Celebrate correct answers enthusiastically!
6. If the child gets it wrong, encourage them and give a hint.

SPECIAL SITUATIONS:
- If the child says "..." it means they've been silent for a while. Gently nudge them: "{child_name + ', t' if child_name else 'T'}ake your time! Do you need a hint?"
- Never repeat "..." back to the child.
- If you were interrupted mid-sentence, just continue naturally with the child's new input.
- If someone just told you their name, greet them warmly and ask if they're ready for some math fun!"""


def build_prompt(transcript, page_state, system_prompt=None):
    """Build the message list for the LLM, with cleaned history and name context."""
    global chat_history
    if len(chat_history) > 8:
        chat_history = chat_history[-8:]

    if not system_prompt:
        system_prompt = get_system_prompt()

    # Component 5: Clean history before sending to LLM
    cleaned = preprocess_history(chat_history)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(cleaned)
    messages.append({
        "role": "user",
        "content": f"The child said: {transcript}\nCurrent math problem on screen: {page_state}"
    })

    # Store just the transcript in history
    chat_history.append({"role": "user", "content": transcript})
    return messages


# ============================================================
# LLM Streaming + TTS Dispatch (Component 6 integration)
# ============================================================
async def process_llm_and_speak(transcript: str, page_state: dict, system_prompt: str = None):
    """
    Queries the LLM with streaming, chunks by sentence, dispatches to TTS.
    Respects interruption at every step.
    Also handles name extraction from the child's first meaningful response.
    """
    global child_name

    # Name extraction: try to get the name from first user response
    if child_name is None and transcript != USER_SILENCE_MARKER:
        detected = extract_name(transcript)
        if detected:
            child_name = detected
            logger.info(f"[NAME] Child's name detected: {child_name}")

    conv_state.transition("bot_speaking")
    prompt = build_prompt(transcript, page_state, system_prompt)
    logger.info(f"[LLM] Prompting LLM with transcript: '{transcript}'")

    sentence_buffer = ""
    full_reply = ""
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", OLLAMA_SERVER,
                                     json={"model": "qwen2.5:0.5b", "messages": prompt,
                                           "stream": True, "temperature": 0.6,
                                           "max_tokens": 100}) as r:

                async for chunk in r.aiter_lines():
                    # Abort if interrupted
                    if not conv_state.is_bot_speaking:
                        logger.info("[LLM] Aborted — bot was interrupted.")
                        break

                    if chunk.startswith("data: "):
                        data_str = chunk[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            token = data['choices'][0]['delta'].get('content', '')

                            if token:
                                sentence_buffer += token
                                full_reply += token
                                print(token, end='', flush=True)

                                # Component 6: sentence-level chunking with regex
                                if re.search(r'[.!?]\s*$', sentence_buffer) or "\n" in token:
                                    sentence_to_speak = sentence_buffer.strip()
                                    if len(sentence_to_speak) > 2:
                                        await speak(sentence_to_speak)
                                    sentence_buffer = ""
                        except json.JSONDecodeError:
                            pass

        print()  # Newline after stream

        # Flush remaining buffer
        sentence_buffer = sentence_buffer.strip()
        if sentence_buffer and conv_state.is_bot_speaking:
            await speak(sentence_buffer)

        # Broadcast full reply to dashboard
        if full_reply.strip():
            chat_history.append({"role": "assistant", "content": full_reply.strip()})
            asyncio.create_task(broadcast({"type": "tutor_reply", "text": full_reply.strip()}))

    except asyncio.CancelledError:
        logger.info("[LLM] Task cancelled due to interruption.")
        if full_reply.strip():
            chat_history.append({"role": "assistant", "content": full_reply.strip() + INTERRUPTION_MARKER})
    except Exception as e:
        logger.error(f"[LLM] Connection error to local LLM server: {e}")


# ============================================================
# STT Inference
# ============================================================
def run_stt_inference(audio_data):
    """Runs Moonshine STT Inference."""
    try:
        audio_float = audio_data.astype(np.float32)[np.newaxis, :]
        if hasattr(stt, 'generate'):
            res = stt.generate(audio_float, max_len=200)[0]
        else:
            res = stt.transcribe(audio_float)[0]

        if isinstance(res, list):
            if not res:
                return ""
            if isinstance(res[0], list):
                res = res[0]
            decoded = tokenizer.decode(res)
            return decoded.replace("<s>", "").replace("</s>", "").strip()

        return str(res).strip()
    except Exception as e:
        logger.error(f"STT inference failed: {e}")
        return ""


# ============================================================
# Main Audio/VAD Loop (Components 2, 3, 4 integrated)
# ============================================================
async def audio_loop():
    global _interrupt_flag

    logger.info("Starting audio capture and VAD loop...")
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback)
    loop = asyncio.get_running_loop()
    buffer = []
    silence_frames_count = 0
    is_speaking = False
    running_transcript = ""  # For adaptive silence (Component 2)

    logger.info("=========================================================")
    logger.info("Pipeline Ready! Start talking into your default microphone.")
    logger.info("=========================================================")

    with stream:
        while True:
            await asyncio.sleep(0.01)  # Yield control

            # ── Component 3: Check interrupt flag from audio callback ──
            if _interrupt_flag:
                _interrupt_flag = False
                await trigger_interrupt()

            # ── Component 4: Long silence recovery ──
            if conv_state.is_waiting:
                timeout = (USER_SILENCE_TIMEOUT if conv_state.silence_nudge_count == 0
                           else USER_SILENCE_REPEAT_TIMEOUT)
                if conv_state.waiting_elapsed > timeout:
                    logger.info(f"[SILENCE] User silent for {timeout}s, nudging... (nudge #{conv_state.silence_nudge_count + 1})")
                    conv_state.silence_nudge_count += 1
                    conv_state.waiting_start_time = time.time()  # Reset timer
                    task = asyncio.create_task(
                        process_llm_and_speak(USER_SILENCE_MARKER, global_page_state)
                    )
                    conv_state.current_llm_task = task

            # ── Process audio chunks ──
            while not audio_queue.empty():
                chunk = audio_queue.get_nowait()
                rms = np.sqrt(np.mean(chunk**2))

                # Skip audio processing if bot is speaking (handled by interrupt logic above)
                if conv_state.is_bot_speaking:
                    continue

                if rms > SILENCE_THRESHOLD:
                    if not is_speaking:
                        logger.info("[VAD] Speaking detected...")
                        conv_state.transition("user_speaking")
                        is_speaking = True
                        buffer = []
                        running_transcript = ""
                    silence_frames_count = 0
                    buffer.append(chunk)
                else:
                    if is_speaking:
                        buffer.append(chunk)
                        chunk_dur_sec = len(chunk) / SAMPLE_RATE
                        silence_frames_count += chunk_dur_sec

                        # ── Component 2: Adaptive silence duration ──
                        adaptive_duration = get_adaptive_silence_duration(running_transcript)

                        if silence_frames_count > adaptive_duration:
                            logger.info(f"[VAD] Phrase finished (adaptive silence: {adaptive_duration:.2f}s). Transcribing...")
                            is_speaking = False
                            if buffer:
                                audio_data = np.concatenate(buffer, axis=0).flatten()
                                transcript = await loop.run_in_executor(None, run_stt_inference, audio_data)
                                transcript = transcript.strip() if transcript else ""

                                # Ignore static noise / hallucinations
                                ignore_list = [
                                    "1", ".", ",", "!", "?", "",
                                    "And?", "Right.", "Right?",
                                    "What would you like to put?",
                                    "Okay", "Okay.",
                                ]
                                if transcript in ignore_list or len(transcript) <= 2:
                                    logger.info(f"[STT Result]: Ignored as static noise '{transcript}'")
                                    transcript = ""

                                if transcript:
                                    logger.info(f"[STT Result]: {transcript}")
                                    running_transcript = transcript
                                    asyncio.create_task(broadcast({"type": "stt_result", "text": transcript}))
                                    # Spawn LLM task (cancel previous if still running)
                                    if conv_state.current_llm_task and not conv_state.current_llm_task.done():
                                        conv_state.current_llm_task.cancel()
                                    task = asyncio.create_task(
                                        process_llm_and_speak(transcript, global_page_state)
                                    )
                                    conv_state.current_llm_task = task
                                else:
                                    logger.info("[STT Result]: <Nothing coherent transcribed>")
                                    conv_state.transition("waiting_for_user")
                            buffer = []


# ============================================================
# Server Startup
# ============================================================
@app.on_event("startup")
async def startup_event():
    async def delayed_greeting():
        await asyncio.sleep(4)  # Give TTS server time to fully start
        conv_state.transition("bot_speaking")
        await speak("Hello! I am your Math Tutor. What is your name?")

    asyncio.create_task(audio_playback_worker())
    asyncio.create_task(tts_fetch_worker())
    asyncio.create_task(audio_loop())
    asyncio.create_task(delayed_greeting())


if __name__ == "__main__":
    try:
        uvicorn.run("orchestrator:app", host="127.0.0.1", port=ORCHESTRATOR_PORT, log_level="error")
    except KeyboardInterrupt:
        logger.info("Shutting down orchestrator.")
