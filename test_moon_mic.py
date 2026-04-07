import numpy as np
import sounddevice as sd
import queue
import time
from moonshine_onnx import MoonshineOnnxModel, load_tokenizer

print("Loading Moonshine STT model...")
stt = MoonshineOnnxModel(model_name="moonshine/tiny")
tokenizer = load_tokenizer()
print("Model loaded successfully!")

audio_queue = queue.Queue()
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.02
SILENCE_DURATION = 1.0

def audio_callback(indata, frames, time_info, status):
    """Callback for sounddevice InputStream to push mic data to the queue."""
    if status:
        print(f"Audio status: {status}")
    audio_queue.put(indata.copy())

def run_stt_inference(audio_data):
    """Runs Moonshine STT Inference."""
    try:
        audio_float = audio_data.astype(np.float32)[np.newaxis, :]
        if hasattr(stt, 'generate'):
            res = stt.generate(audio_float, max_len=200)[0]
        else:
            res = stt.transcribe(audio_float)[0]
            
        if isinstance(res, list):
            if not res: return ""
            if isinstance(res[0], list):
                res = res[0]
            decoded = tokenizer.decode(res)
            return decoded.replace("<s>", "").replace("</s>", "").strip()
        return str(res).strip()
    except Exception as e:
        print(f"STT inference failed: {e}")
        return ""

def main():
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback)
    buffer = []
    silence_frames_count = 0
    is_speaking = False
    
    print("=========================================================")
    print("🎙️ Moonshine STT Standalone Test")
    print("Speak into your default microphone. Press Ctrl+C to stop.")
    print("=========================================================")
    
    try:
        with stream:
            while True:
                time.sleep(0.01)
                while not audio_queue.empty():
                    chunk = audio_queue.get_nowait()
                    rms = np.sqrt(np.mean(chunk**2))
                    if rms > SILENCE_THRESHOLD:
                        if not is_speaking:
                            print("\n[VAD] Speaking detected...", end="", flush=True)
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
                                print(" finished. Transcribing...")
                                is_speaking = False
                                if buffer:
                                    audio_data = np.concatenate(buffer, axis=0).flatten()
                                    start_time = time.time()
                                    transcript = run_stt_inference(audio_data)
                                    inference_time = time.time() - start_time
                                    
                                    # Ignore tiny noises
                                    if transcript not in ["1", ".", ",", "!", "?", ""]:
                                        print(f"👉 [STT Result]: '{transcript}' (took {inference_time:.3f}s)")
                                    else:
                                        print(f"   (Ignored static noise: '{transcript}')")
                                buffer = []
    except KeyboardInterrupt:
        print("\n\nStopped testing.")

if __name__ == "__main__":
    main()