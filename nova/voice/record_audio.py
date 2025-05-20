import sounddevice as sd
import soundfile as sf

def record_audio(filename: str, duration: int = 5, samplerate: int = 44100):
    print(f"🎙️ Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    sf.write(filename, audio, samplerate)
    print(f"✅ Saved audio to {filename}")
