import whisper

model = whisper.load_model("base")  # You can use "small", "medium", etc.

def transcribe(audio_path: str) -> str:
    print(f"🧠 Transcribing {audio_path}...")
    result = model.transcribe(audio_path)
    return result["text"].strip()
