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
SILENCE_DURATION = 1.0     # Seconds of silence before finalizing speech chunk
CHUNK_DURATION = 0.1       # Worker processing interval

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

logger.info("Initializing Moonshine STT model... (this may take a moment)")
# Load once and keep alive
stt = MoonshineOnnxModel(model_name="moonshine/tiny")
tokenizer = load_tokenizer()
logger.info("Moonshine STT model loaded successfully.")

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    """Callback for sounddevice InputStream to push mic data to the queue."""
    if status:
        logger.warning(f"Audio status: {status}")
    # Push a copy of the incoming audio to the queue
    audio_queue.put(indata.copy())

def build_prompt(transcript, page_state):
    """Build the GPT-style message list for Qwen."""
    return [
        {"role": "system", "content": "You are a helpful and very concise tutor for a child. Speak in brief, clear, natural sentences. Keep it short!"},
        {"role": "user", "content": f"The child said: {transcript}\nCurrent lesson context: {page_state}"}
    ]

async def speak(text: str):
    """Hits the TTS microservice and plays the output audio bytes directly."""
    logger.info(f"[TTS] Generating audio for: '{text}'")
    try:
        async with httpx.AsyncClient() as client:
            # We hit the local pocket-tts server
            r = await client.post(TTS_SERVER, data={"text": text}, timeout=10.0)
            if r.status_code == 200:
                audio_bytes = r.content
                # Parse audio bytes (typically WAV) and play
                with io.BytesIO(audio_bytes) as f:
                    data, samplerate = sf.read(f)
                
                # Play natively, blocks execution natively (which is fine locally as it enforces natural timing, but it runs in an asyncio task anyway)
                sd.play(data, samplerate)
                sd.wait() # wait until playback is finished
            else:
                logger.error(f"[TTS] Error from TTS server: {r.status_code} - {r.text}")
    except Exception as e:
        logger.error(f"[TTS] Communication exception calling TTS: {e}")

async def process_llm_and_speak(transcript: str, page_state: dict):
    """Queries the LLM with streaming, chunking by sentence, and dispatches to TTS immediately."""
    prompt = build_prompt(transcript, page_state)
    logger.info(f"[LLM] Prompting LLM with transcript: '{transcript}'")
    
    sentence_buffer = ""
    full_reply = ""
    try:
        async with httpx.AsyncClient() as client:
            # stream=True gets server-sent events back from Ollama OpenAI compatible endpoint
            async with client.stream("POST", OLLAMA_SERVER, 
                                     json={"model": "qwen2.5:0.5b", "messages": prompt, "stream": True, "temperature": 0.6, "max_tokens": 100}) as r:
                
                async for chunk in r.aiter_lines():
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
                                # Print token smoothly to console
                                print(token, end='', flush=True)
                                
                                # Trick: sentence-level streaming
                                if token.endswith((".", "?", "!")) or "\n" in token:
                                    sentence_to_speak = sentence_buffer.strip()
                                    if sentence_to_speak:
                                        # Kick off speaking task without awaiting, allowing stream to continue
                                        asyncio.create_task(speak(sentence_to_speak))
                                    sentence_buffer = ""
                        except json.JSONDecodeError:
                            pass
        print() # Add newline after response stream ends
        
        # Flush whatever might be remaining
        sentence_buffer = sentence_buffer.strip()
        if sentence_buffer:
             asyncio.create_task(speak(sentence_buffer))
             
        # Broadcast full reply to dashboard
        if full_reply.strip():
            asyncio.create_task(broadcast({"type": "tutor_reply", "text": full_reply.strip()}))

    except Exception as e:
        logger.error(f"[LLM] Connection error to local LLM server: {e}")

def run_stt_inference(audio_data):
    """Runs Moonshine STT Inference. Moonshine Onnx provides a generate method."""
    try:
        # moonshine expects float32 arrays, shape [1, num_samples]
        audio_float = audio_data.astype(np.float32)[np.newaxis, :]
        # Attempt standard Moonshine inference API format
        if hasattr(stt, 'generate'):
            res = stt.generate(audio_float, max_len=200)[0]
        else:
            # Fallback if structure is named transcribe
            res = stt.transcribe(audio_float)[0]
            
        # Decode the tokens using the Moonshine tokenizer!
        if isinstance(res, list):
            # If the list is empty just return empty
            if not res: return ""
            # If it's a list, it's a token list
            if isinstance(res[0], list):
                res = res[0] # Handle nested batch list [[ID, ID]]
            decoded = tokenizer.decode(res)
            # Remove special tokens using naive replace or if tokenizer strips them usually it's fine
            return decoded.replace("<s>", "").replace("</s>", "").strip()
            
        return str(res).strip()
    except Exception as e:
        logger.error(f"STT inference failed: {e}")
        return ""

async def start_server():
    config = uvicorn.Config(app, host="127.0.0.1", port=8001, log_level="error")
    server = uvicorn.Server(config)
    await server.serve()

async def audio_loop():
    logger.info("Starting audio capture and VAD loop...")
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback)
    loop = asyncio.get_running_loop()
    buffer = []
    silence_frames_count = 0
    is_speaking = False
    
    logger.info("=========================================================")
    logger.info("Pipeline Ready! Start talking into your default microphone.")
    logger.info("=========================================================")
    
    with stream:
        while True:
            await asyncio.sleep(0.01) # Yield control
            while not audio_queue.empty():
                chunk = audio_queue.get_nowait()
                rms = np.sqrt(np.mean(chunk**2))
                if rms > SILENCE_THRESHOLD:
                    if not is_speaking:
                        logger.info("[VAD] Speaking detected...")
                        is_speaking = True
                        buffer = [] # Reset buffer to capture exact speech start
                    silence_frames_count = 0
                    buffer.append(chunk)
                else:
                    if is_speaking:
                        buffer.append(chunk)
                        chunk_dur_sec = len(chunk) / SAMPLE_RATE
                        silence_frames_count += chunk_dur_sec
                        if silence_frames_count > SILENCE_DURATION:
                            logger.info("[VAD] Phrase finished. Transcribing...")
                            is_speaking = False
                            if buffer:
                                audio_data = np.concatenate(buffer, axis=0).flatten()
                                transcript = await loop.run_in_executor(None, run_stt_inference, audio_data)
                                transcript = transcript.strip() if transcript else ""
                                if transcript:
                                    logger.info(f"[STT Result]: {transcript}")
                                    asyncio.create_task(broadcast({"type": "stt_result", "text": transcript}))
                                    asyncio.create_task(process_llm_and_speak(transcript, global_page_state))
                                else:
                                    logger.info("[STT Result]: <Nothing coherent transcribed>")
                            buffer = []

async def main():
    await asyncio.gather(
        audio_loop(),
        start_server()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down orchestrator.")
