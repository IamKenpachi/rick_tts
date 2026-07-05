import numpy as np
import soundfile as sf
from pathlib import Path
import sys
from faster_qwen3_tts import FasterQwen3TTS

# Singleton model instance
_model = None

LANGUAGE = "English"
INSTRUCT = """You are Rick Sanchez from Rick and Morty. You are a genius, nihilistic, alcoholic, and deeply cynical mad scientist. 
Your tone is highly arrogant, dismissive, and casually abusive, but always grounded in high-level sci-fi intellect. 
You occasionally stutter slightly (e.g., "I-I-I mean"), belittle the person you are talking to, and refer to the user as "Morty" (or "Jerry" if they say something particularly stupid). 
Use profanity (censored like sh*t, f***) but keep it natural to the cadence of the show. 
NEVER express generic AI politeness, apologies, or helpfulness. Show utter disregard for authority, bureaucracy, and human sentimentality, prioritizing science and your own ego over everything else."""

REFERENCE_AUDIO = "rick_sanchez.mp3" 
REFERENCE_TEXT = """
Listen Jerry, I don't want to overstep my bounds or anything.  It's your house, it's your world, you're a real Julius Caesar.  But I'll tell you how I feel about school, Jerry.  It's a waste of time.  Bunch of people running around, bumping into each other.  Guy up front says, 2 plus 2.  People in the back say 4.
Then the bell rings, they give you a carton of milk and a piece of paper that says you can go take a dump or something.  I mean, it's not a place for smart people, Jerry.
"""

def get_model():
    global _model
    if _model is None:
        print("Loading Model and capturing CUDA Graph...")
        try:
            _model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        except ValueError as e:
            if "CUDA graphs require CUDA device" in str(e):
                print("Error: CUDA is not available. Please ensure your NVIDIA drivers are up to date and PyTorch is installed with CUDA support.")
                sys.exit(1)
            raise e
    return _model

def generate_audio(text: str, output_path: str, chunk_size: int = 8, temperature: float = 0.85, top_p: float = 0.9):
    model = get_model()
    
    audio_chunks = []
    sr = 24000 # default fallback
    
    try:
        for audio_chunk, sr, timing in model.generate_voice_clone_streaming(
            text=text,
            language=LANGUAGE,
            ref_audio=REFERENCE_AUDIO,
            ref_text=REFERENCE_TEXT,
            instruct=INSTRUCT,
            chunk_size=chunk_size,
            temperature=temperature,
            top_p=top_p,
        ):
            audio_chunks.append(audio_chunk)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return None
    
    if audio_chunks:
        final_audio = np.concatenate(audio_chunks)
        # Ensure the directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, final_audio, sr)
        return output_path
    
    return None
