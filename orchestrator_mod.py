import asyncio
import httpx
import json
import logging
import queue
import time
import io

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

# Pipeline Configuration
OLLAMA_SERVER = "http://localhost:11434/v1/chat/completions"
TTS_SERVER = "http://localhost:8000/tts"

# Audio Settings
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.02   # Simple RMS amplitude threshold for VAD
SILENCE_DURATION = 0.5     # Seconds of silence before finalizing speech chunk

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

class AppState:
    conversation_state = "waiting_for_user"  # waiting_for_user, user_speaking, bot_speaking
    chat_history = []
    
    # Queues for background tasks
    tts_text_queue = asyncio.Queue()
    audio_playback_queue = asyncio.Queue()
    
    # Task references for cancellation
    llm_task = None
    
state = AppState()

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

logger.info("Initializing Moonshine STT model...")
stt = MoonshineOnnxModel(model_name="moonshine/base")
tokenizer = load_tokenizer()
logger.info("Moonshine STT model loaded successfully.")

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        logger.warning(f"Audio status: {status}")
    if state.conversation_state == "bot_speaking":
        # Check against interruption threshold
        rms = np.sqrt(np.mean(indata**2))
        if rms > SILENCE_THRESHOLD * 2: # Give it a slightly higher threshold when bot is speaking to avoid self-interruption from speaker echo
            # We must use call_soon_threadsafe because this is a C callback
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(interrupt_bot)
            audio_queue.put(indata.copy())
    else:
        audio_queue.put(indata.copy())

def interrupt_bot():
    if state.conversation_state == "bot_speaking":
        logger.info("[VAD] User interrupted! Stopping bot...")
        sd.stop() # Immediately stop current sounddevice playback
        
        # Cancel LLM generation if running
        if state.llm_task and not state.llm_task.done():
            state.llm_task.cancel()
            
        # Clear queues
        while not state.tts_text_queue.empty():
            state.tts_text_queue.get_nowait()
        while not state.audio_playback_queue.empty():
            state.audio_playback_queue.get_nowait()
            
        state.conversation_state = "waiting_for_user"

def build_prompt(transcript, page_state):
    """Advanced Unmute-like system prompt"""
    system_prompt = """You are an interactive, extremely concise and encouraging math tutor for a child.
RULES:
1. Speak in brief, natural sentences.
2. Stop at 1 or 2 sentences max.
3. Be conversational.
4. Current Screen/Math problem Context: {page_context}
""".format(page_context=json.dumps(page_state))

    messages = [{"role": "system", "content": system_prompt}]
    
    # Maintain short rolling history (last 4 turns)
    if len(state.chat_history) > 4:
        state.chat_history = state.chat_history[-4:]
        
    messages.extend(state.chat_history)
    messages.append({"role": "user", "content": transcript})
    
    # Update local history
    state.chat_history.append({"role": "user", "content": transcript})
    return messages

async def audio_playback_worker():
    while True:
        data, samplerate = await state.audio_playback_queue.get()
        if state.conversation_state != "bot_speaking":
            state.audio_playback_queue.task_done()
            continue
            
        def _play():
            try:
                sd.play(data, samplerate)
                sd.wait()
            except sd.PortAudioError:
                pass # Playback aborted by interrupt
                
        await asyncio.to_thread(_play)
        
        # After finishing queue, if nothing else is pending, revert state
        if state.audio_playback_queue.empty() and state.tts_text_queue.empty() and (not state.llm_task or state.llm_task.done()):
            if state.conversation_state == "bot_speaking":
                state.conversation_state = "waiting_for_user"
                
        state.audio_playback_queue.task_done()

async def tts_fetch_worker():
    while True:
        text = await state.tts_text_queue.get()
        if state.conversation_state != "bot_speaking":
            state.tts_text_queue.task_done()
            continue
            
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(TTS_SERVER, data={"text": text}, timeout=10.0)
                if r.status_code == 200:
                    with io.BytesIO(r.content) as f:
                        data, samplerate = sf.read(f)
                    await state.audio_playback_queue.put((data, samplerate))
        except Exception as e:
            logger.error(f"[TTS Worker] Error: {e}")
        state.tts_text_queue.task_done()

async def process_llm_task(transcript: str, page_state: dict):
    state.conversation_state = "bot_speaking"
    prompt = build_prompt(transcript, page_state)
    logger.info(f"[LLM] Dispatching prompt...")
    
    sentence_buffer = ""
    full_reply = ""
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", OLLAMA_SERVER, 
                                     json={"model": "qwen2.5:0.5b", "messages": prompt, "stream": True, "temperature": 0.6, "max_tokens": 100}) as r:
                async for chunk in r.aiter_lines():
                    if state.conversation_state != "bot_speaking":
                        break # Abort generation if interrupted
                        
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
                                
                                if token.endswith((".", "?", "!", ",")) or "\n" in token:
                                    s = sentence_buffer.strip()
                                    if len(s) > 2:
                                        await state.tts_text_queue.put(s)
                                    sentence_buffer = ""
                        except json.JSONDecodeError:
                            pass
                            
        print() 
        s = sentence_buffer.strip()
        if s and state.conversation_state == "bot_speaking":
            await state.tts_text_queue.put(s)
            
        if full_reply.strip():
            state.chat_history.append({"role": "assistant", "content": full_reply.strip()})
            await broadcast({"type": "tutor_reply", "text": full_reply.strip()})
            
    except asyncio.CancelledError:
        logger.info("[LLM] Task cancelled due to interruption.")
    except Exception as e:
        logger.error(f"[LLM] Error: {e}")

def run_stt_inference(audio_data):
    try:
        audio_float = audio_data.astype(np.float32)[np.newaxis, :]
        if hasattr(stt, 'generate'):
            res = stt.generate(audio_float, max_len=200)[0]
        else:
            res = stt.transcribe(audio_float)[0]
            
        if isinstance(res, list):
            if not res: return ""
            if isinstance(res[0], list): res = res[0]
            decoded = tokenizer.decode(res)
            return decoded.replace("<s>", "").replace("</s>", "").strip()
        return str(res).strip()
    except Exception as e:
        return ""

async def audio_loop():
    logger.info("Starting VAD loop...")
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback)
    loop = asyncio.get_running_loop()
    buffer = []
    silence_frames_count = 0
    is_speaking = False
    
    with stream:
        while True:
            await asyncio.sleep(0.01)
            while not audio_queue.empty():
                chunk = audio_queue.get_nowait()
                rms = np.sqrt(np.mean(chunk**2))
                
                # VAD ignores mic input completely if bot is speaking AND voice isn't loud enough to interrupt
                if state.conversation_state == "bot_speaking":
                    continue
                    
                if rms > SILENCE_THRESHOLD:
                    if not is_speaking:
                        state.conversation_state = "user_speaking"
                        is_speaking = True
                        buffer = []
                    silence_frames_count = 0
                    buffer.append(chunk)
                else:
                    if is_speaking:
                        buffer.append(chunk)
                        chunk_dur_sec = len(chunk) / SAMPLE_RATE
                        silence_frames_count += chunk_dur_sec
                        if silence_frames_count > SILENCE_DURATION:
                            is_speaking = False
                            if buffer:
                                audio_data = np.concatenate(buffer, axis=0).flatten()
                                transcript = await loop.run_in_executor(None, run_stt_inference, audio_data)
                                transcript = transcript.strip() if transcript else ""
                                
                                ignore_list = ["1", ".", ",", "!", "?", "", "And?", "Right.", "Okay"]
                                if transcript in ignore_list or len(transcript) <= 2:
                                    transcript = ""
                                    
                                if transcript:
                                    logger.info(f"[STT User Said]: {transcript}")
                                    await broadcast({"type": "stt_result", "text": transcript})
                                    # Spawn LLM Task
                                    if state.llm_task and not state.llm_task.done():
                                        state.llm_task.cancel()
                                    state.llm_task = asyncio.create_task(process_llm_task(transcript, global_page_state))
                                else:
                                    state.conversation_state = "waiting_for_user"
                            buffer = []

async def start_server():
    config = uvicorn.Config(app, host="127.0.0.1", port=8001, log_level="error")
    server = uvicorn.Server(config)
    await server.serve()

async def main():
    asyncio.create_task(audio_playback_worker())
    asyncio.create_task(tts_fetch_worker())
    await asyncio.gather(
        audio_loop(),
        start_server()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down orchestrator.")
