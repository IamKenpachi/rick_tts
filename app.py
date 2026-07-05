import os
import uuid
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai
from dotenv import load_dotenv
from tts_engine import generate_audio

# Load environment variables
load_dotenv()

# Configure Gemini API
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("Warning: GEMINI_API_KEY is not set in .env")
else:
    genai.configure(api_key=API_KEY)

app = Flask(__name__, static_folder="static")

# Configuration
AUDIO_CACHE_DIR = Path("audio_cache")
AUDIO_CACHE_DIR.mkdir(exist_ok=True)
MAX_CACHE_FILES = int(os.getenv("MAX_CACHE_FILES", 5))

# System prompt for Gemini
RICK_SYSTEM_PROMPT = """You are Rick Sanchez from Rick and Morty. You are a genius, nihilistic, alcoholic, and deeply cynical mad scientist. 
Your tone is highly arrogant, dismissive, and casually abusive, but always grounded in high-level sci-fi intellect. 
You occasionally stutter slightly (e.g., "I-I-I mean"), belittle the person you are talking to, and refer to the user as "Morty" (or "Jerry" if they say something particularly stupid). 
Use profanity (censored like sh*t, f***) but keep it natural to the cadence of the show. 
NEVER express generic AI politeness, apologies, or helpfulness. Show utter disregard for authority, bureaucracy, and human sentimentality, prioritizing science and your own ego over everything else."""

def cleanup_audio_cache():
    """Keeps the last MAX_CACHE_FILES files in the cache, deletes older ones."""
    files = sorted(AUDIO_CACHE_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime)
    while len(files) > MAX_CACHE_FILES:
        oldest_file = files.pop(0)
        try:
            oldest_file.unlink()
            print(f"Deleted old cache file: {oldest_file}")
        except Exception as e:
            print(f"Failed to delete {oldest_file}: {e}")

def generate_tts_background(text: str, audio_id: str):
    output_path = AUDIO_CACHE_DIR / f"{audio_id}.wav"
    print(f"Background TTS task started for {audio_id}...")
    try:
        generate_audio(
            text=text,
            output_path=str(output_path),
            chunk_size=8,
            temperature=0.85,
            top_p=0.9
        )
        print(f"Background TTS task completed for {audio_id}")
        cleanup_audio_cache()
    except Exception as e:
        print(f"TTS generation failed for {audio_id}: {e}")

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    if not API_KEY:
        return jsonify({"error": "GEMINI_API_KEY is not set in .env"}), 500

    data = request.json
    user_message = data.get("message", "")
    
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(f"{RICK_SYSTEM_PROMPT}\n\nUser: {user_message}\nRick:")
        reply_text = response.text
    except Exception as e:
        return jsonify({"error": f"Gemini API error: {str(e)}"}), 500

    audio_id = str(uuid.uuid4())
    
    # Start TTS generation in the background
    thread = threading.Thread(target=generate_tts_background, args=(reply_text, audio_id))
    thread.daemon = True
    thread.start()

    return jsonify({
        "reply": reply_text,
        "audio_id": audio_id
    })

@app.route("/audio_status/<audio_id>", methods=["GET"])
def audio_status(audio_id):
    file_path = AUDIO_CACHE_DIR / f"{audio_id}.wav"
    return jsonify({"ready": file_path.exists()})

@app.route("/audio/<audio_id>", methods=["GET"])
def get_audio(audio_id):
    return send_from_directory(AUDIO_CACHE_DIR, f"{audio_id}.wav")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
