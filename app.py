import os
import uuid
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from google import genai as google_genai
from dotenv import load_dotenv
import re
import time
from collections import defaultdict

try:
    from tts_engine import generate_audio, warmup_model
    TTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TTS engine not available: {e}")
    TTS_AVAILABLE = False

TTS_READY = False  # becomes True when warmup_model() finishes successfully
_tts_semaphore = threading.Semaphore(1)  # Only 1 TTS job at a time (model is not thread-safe)

# Load environment variables
load_dotenv()

# Configure Gemini API
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("Warning: GEMINI_API_KEY is not set in .env")
    _genai_client = None
else:
    _genai_client = google_genai.Client(api_key=API_KEY)

app = Flask(__name__, static_folder="static")

# Configuration
AUDIO_CACHE_DIR = Path(__file__).parent / "audio_cache"
AUDIO_CACHE_DIR.mkdir(exist_ok=True)
MAX_CACHE_FILES = int(os.getenv("MAX_CACHE_FILES", 5))

UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

_rate_limit_store = defaultdict(float)  # ip -> last request timestamp
RATE_LIMIT_SECONDS = float(os.getenv("RATE_LIMIT_SECONDS", 5.0))  # min seconds between requests per IP

def _warmup_tts_background():
    """Warms up the TTS model in a background thread on server start."""
    global TTS_READY
    if TTS_AVAILABLE:
        print("Starting TTS model warm-up in background thread...")
        success = warmup_model()
        TTS_READY = success
        if success:
            print("TTS model is ready. First user request will not wait for model loading.")
        else:
            print("TTS warm-up failed. First request will attempt to load the model.")
    else:
        print("TTS not available. Skipping warm-up.")

_warmup_thread = threading.Thread(target=_warmup_tts_background, daemon=True)
_warmup_thread.start()

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
    global TTS_READY
    if not TTS_AVAILABLE:
        print("TTS Engine unavailable. Skipping TTS generation.")
        return

    output_path = AUDIO_CACHE_DIR / f"{audio_id}.wav"
    print(f"Background TTS task started for {audio_id}...")
    
    with _tts_semaphore:  # CRITICAL: Only one TTS job may run at a time
        try:
            generate_audio(
                text=text,
                output_path=str(output_path),
                chunk_size=int(os.getenv("TTS_CHUNK_SIZE", 8)),
                temperature=float(os.getenv("TTS_TEMPERATURE", 0.85)),
                top_p=float(os.getenv("TTS_TOP_P", 0.9))
            )
            print(f"Background TTS task completed for {audio_id}")
            cleanup_audio_cache()
        except Exception as e:
            print(f"TTS generation failed for {audio_id}: {e}")

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/tts_status", methods=["GET"])
def tts_status():
    """Returns whether the TTS model has finished warming up."""
    return jsonify({"ready": TTS_READY, "available": TTS_AVAILABLE})

@app.route("/chat", methods=["POST"])
def chat():
    # Rate limiting
    client_ip = request.remote_addr
    now = time.time()
    if now - _rate_limit_store[client_ip] < RATE_LIMIT_SECONDS:
        return jsonify({"error": f"Too many requests. Please wait {RATE_LIMIT_SECONDS:.0f}s between messages."}), 429
    _rate_limit_store[client_ip] = now

    if not API_KEY:
        return jsonify({"error": "GEMINI_API_KEY is not set in .env"}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request must be JSON with Content-Type: application/json"}), 400
    user_message = data.get("message", "")
    
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    try:
        if not _genai_client:
            return jsonify({"error": "Gemini Client not initialized"}), 500
        
        response = _genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{RICK_SYSTEM_PROMPT}\n\nUser: {user_message}\nRick:"
        )
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
    if not UUID_PATTERN.match(audio_id):
        return jsonify({"error": "Invalid audio_id"}), 400
    file_path = AUDIO_CACHE_DIR / f"{audio_id}.wav"
    return jsonify({"ready": file_path.exists()})

@app.route("/audio/<audio_id>", methods=["GET"])
def get_audio(audio_id):
    if not UUID_PATTERN.match(audio_id):
        return jsonify({"error": "Invalid audio_id"}), 400
    return send_from_directory(AUDIO_CACHE_DIR, f"{audio_id}.wav")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
