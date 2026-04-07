import re
with open("orchestrator.py", "r") as f:
    content = f.read()

# Replace speak function and add queues
new_speak_and_queues = """
tts_text_queue = asyncio.Queue()
audio_playback_queue = asyncio.Queue()

async def audio_playback_worker():
    global is_tts_playing
    while True:
        data, samplerate = await audio_playback_queue.get()
        is_tts_playing = True
        
        # Clear microphone queue to prevent hearing itself
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break
                
        def _play():
            import time
            sd.play(data, samplerate)
            sd.wait()
            time.sleep(0.3)
            
        await asyncio.to_thread(_play)
        
        is_tts_playing = False
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break
                
        audio_playback_queue.task_done()

async def tts_fetch_worker():
    while True:
        text = await tts_text_queue.get()
        logger.info(f"[TTS Worker] Fetching audio for: '{text}'")
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(TTS_SERVER, data={"text": text}, timeout=15.0)
                if r.status_code == 200:
                    audio_bytes = r.content
                    with io.BytesIO(audio_bytes) as f:
                        data, samplerate = sf.read(f)
                    await audio_playback_queue.put((data, samplerate))
                else:
                    logger.error(f"[TTS] Error from TTS server: {r.status_code}")
        except Exception as e:
            logger.error(f"[TTS Worker] Communication exception: {e}")
        tts_text_queue.task_done()

async def speak(text: str):
    \"\"\"Queues text to be processed by the TTS worker.\"\"\"
    await tts_text_queue.put(text)
"""

content = re.sub(r'async def speak\(text: str\):.*?except Exception as e:\n        logger.error\(f"\[TTS\] Communication exception calling TTS: \{e\}"\)', new_speak_and_queues, content, flags=re.DOTALL)

# Update startup event to launch the workers
new_startup = """@app.on_event("startup")
async def startup_event():
    async def delayed_greeting():
        await asyncio.sleep(4) # Give TTS server time to fully start
        await speak("Hello! I am your Math Tutor. What is your name? Let's solve some math problems together!")
        
    asyncio.create_task(audio_playback_worker())
    asyncio.create_task(tts_fetch_worker())
    asyncio.create_task(audio_loop())
    asyncio.create_task(delayed_greeting())"""

content = re.sub(r'@app.on_event\("startup"\).*?asyncio.create_task\(delayed_greeting\(\)\)', new_startup, content, flags=re.DOTALL)

with open("orchestrator.py", "w") as f:
    f.write(content)
print("Updated orchestrator.py")
