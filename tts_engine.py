import numpy as np
import soundfile as sf
import os
import librosa
from pathlib import Path
from faster_qwen3_tts import FasterQwen3TTS

_model = None
_current_model_id = None
_cached_voice_clone_prompt = None
SUPPORTED_MODELS = {
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}
DEFAULT_MODEL_SIZE = "1.7B"

LANGUAGE = "English"
INSTRUCT = """You are Rick Sanchez from Rick and Morty. You are a genius, nihilistic, alcoholic, and deeply cynical mad scientist. 
Your tone is highly arrogant, dismissive, and casually abusive, but always grounded in high-level sci-fi intellect. 
You occasionally stutter slightly (e.g., "I-I-I mean"), belittle the person you are talking to, and refer to the user as "Morty" (or "Jerry" if they say something particularly stupid). 
Use profanity (censored like sh*t, f***) but keep it natural to the cadence of the show. 
NEVER express generic AI politeness, apologies, or helpfulness. Show utter disregard for authority, bureaucracy, and human sentimentality, prioritizing science and your own ego over everything else."""

REFERENCE_AUDIO = str(Path(__file__).parent / "rick_sanchez.mp3")
REFERENCE_TEXT = """
Listen Jerry, I don't want to overstep my bounds or anything.  It's your house, it's your world, you're a real Julius Caesar.  But I'll tell you how I feel about school, Jerry.  It's a waste of time.  Bunch of people running around, bumping into each other.  Guy up front says, 2 plus 2.  People in the back say 4.
Then the bell rings, they give you a carton of milk and a piece of paper that says you can go take a dump or something.  I mean, it's not a place for smart people, Jerry.
"""

def get_model(model_id: str = None):
    global _model, _current_model_id
    import torch

    if model_id is None:
        model_id = _current_model_id or SUPPORTED_MODELS[DEFAULT_MODEL_SIZE]

    if _model is not None and _current_model_id != model_id:
        print(f"Unloading model {_current_model_id} to load {model_id}...")
        del _model
        _model = None
        _current_model_id = None
        torch.cuda.empty_cache()
        print("Old model unloaded. GPU memory freed.")

    if _model is None:
        print(f"Loading model {model_id} and capturing CUDA Graph...")
        try:
            _model = FasterQwen3TTS.from_pretrained(model_id)
            _current_model_id = model_id
        except ValueError as e:
            if "CUDA graphs require CUDA device" in str(e):
                raise RuntimeError(
                    "CUDA is not available. Ensure your NVIDIA drivers are up to date and "
                    "PyTorch is installed with CUDA support. Cannot run FasterQwen3TTS without GPU."
                ) from e
            raise e
    return _model

def get_current_model_id() -> str:
    return _current_model_id or SUPPORTED_MODELS[DEFAULT_MODEL_SIZE]

def get_cached_voice_clone_prompt():
    return _cached_voice_clone_prompt

def warmup_model(model_id: str = None):
    global _cached_voice_clone_prompt
    try:
        model = get_model(model_id)
        print("TTS warm-up: model loaded. Running silent inference pass...")
        audio_list, sr = model.generate_voice_clone(
            text="Ready.",
            language=LANGUAGE,
            ref_audio=REFERENCE_AUDIO,
            ref_text=REFERENCE_TEXT,
            instruct=INSTRUCT,
            temperature=float(os.getenv("TTS_TEMPERATURE", 0.85)),
            top_p=float(os.getenv("TTS_TOP_P", 0.9)),
            top_k=int(os.getenv("TTS_TOP_K", 40)),
            repetition_penalty=float(os.getenv("TTS_REPETITION_PENALTY", 1.05)),
        )
        # Cache the prompt
        cache_key = (str(REFERENCE_AUDIO), REFERENCE_TEXT)
        if hasattr(model, "_voice_prompt_cache") and cache_key in model._voice_prompt_cache:
            _cached_voice_clone_prompt = model._voice_prompt_cache[cache_key]
            print(f"Voice clone prompt cached successfully.")
        else:
            print("Warning: voice clone prompt cache not found; will re-encode on each request.")
            _cached_voice_clone_prompt = None
        print("TTS warm-up complete. Model is ready.")
        return True
    except Exception as e:
        print(f"TTS warm-up failed: {e}")
        return False

def generate_audio(
    text: str,
    output_path: str,
    chunk_size: int = 8,
    temperature: float = 0.85,
    top_p: float = 0.9,
    top_k: int = 40,
    repetition_penalty: float = 1.05,
    use_streaming: bool = False,
    model_id: str = None,
):
    model = get_model(model_id)
    cached_prompt = _cached_voice_clone_prompt

    if use_streaming:
        audio_chunks = []
        sr = 24000
        gen_kwargs = dict(
            text=text,
            language=LANGUAGE,
            instruct=INSTRUCT,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            chunk_size=chunk_size,
        )
        if cached_prompt is not None:
            gen_kwargs["voice_clone_prompt"] = cached_prompt
        else:
            gen_kwargs["ref_audio"] = REFERENCE_AUDIO
            gen_kwargs["ref_text"] = REFERENCE_TEXT
        for audio_chunk, sr, timing in model.generate_voice_clone_streaming(**gen_kwargs):
            audio_chunks.append(audio_chunk)
        if not audio_chunks:
            return None
        final_audio = np.concatenate(audio_chunks)
    else:
        gen_kwargs = dict(
            text=text,
            language=LANGUAGE,
            instruct=INSTRUCT,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        if cached_prompt is not None:
            gen_kwargs["voice_clone_prompt"] = cached_prompt
        else:
            gen_kwargs["ref_audio"] = REFERENCE_AUDIO
            gen_kwargs["ref_text"] = REFERENCE_TEXT
        audio_list, sr = model.generate_voice_clone(**gen_kwargs)
        if not audio_list:
            return None
        final_audio = np.concatenate(audio_list)

    # Apply speed factor BEFORE writing
    speed_factor = float(os.getenv("TTS_SPEED_FACTOR", 1.0))
    if speed_factor != 1.0:
        stretch_rate = 1.0 / speed_factor
        final_audio = librosa.effects.time_stretch(final_audio.astype(np.float32), rate=stretch_rate)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, final_audio, sr)
    return output_path

