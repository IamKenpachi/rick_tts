import numpy as np
import soundfile as sf
import os
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
    global _model, _current_model_id, _cached_voice_clone_prompt
    import torch

    if model_id is None:
        model_id = _current_model_id or SUPPORTED_MODELS[DEFAULT_MODEL_SIZE]

    if _model is not None and _current_model_id != model_id:
        print(f"Unloading model {_current_model_id} to load {model_id}...")
        del _model
        _model = None
        _current_model_id = None
        _cached_voice_clone_prompt = None
        torch.cuda.empty_cache()
        print("Old model unloaded. GPU memory freed.")

    if _model is None:
        print(f"Loading model {model_id} and capturing CUDA Graph...")
        
        # Monkey patch Qwen3TTSModel to prevent device_map="cuda" from causing meta tensor errors
        # This is a known issue with the 0.6B model on some PyTorch/accelerate combinations.
        try:
            from qwen_tts import Qwen3TTSModel
            orig_from_pretrained = Qwen3TTSModel.from_pretrained
            
            @classmethod
            def custom_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
                device = kwargs.pop("device_map", None)
                # Load fully into CPU first to avoid tied-weight meta tensor errors
                wrapper = orig_from_pretrained.__func__(cls, pretrained_model_name_or_path, *model_args, **kwargs)
                if device:
                    wrapper.model = wrapper.model.to(device)
                    # Qwen3TTSModel caches device in self.device, so we must update it
                    wrapper.device = next(wrapper.model.parameters()).device
                return wrapper
                
            Qwen3TTSModel.from_pretrained = custom_from_pretrained
            patched = True
        except ImportError:
            patched = False

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
        finally:
            if patched:
                Qwen3TTSModel.from_pretrained = orig_from_pretrained
    return _model

def get_current_model_id() -> str:
    return _current_model_id or SUPPORTED_MODELS[DEFAULT_MODEL_SIZE]

def get_cached_voice_clone_prompt():
    return _cached_voice_clone_prompt

def warmup_model(model_id: str = None):
    global _cached_voice_clone_prompt
    try:
        model = get_model(model_id)
        print("TTS warm-up: model loaded.")
        
        if not getattr(model, "_warmed_up", False):
            print("Running silent inference pass to capture CUDA graphs...")
            model.generate_voice_clone(
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
        else:
            print("CUDA graphs already captured. Pre-encoding reference audio for cache...")
            dummy_input_ids = [model.model._tokenize_texts(
                [model.model._build_assistant_text("OK.")]
            )[0]]
            model._resolve_voice_clone_prompt_from_reference(
                input_ids=dummy_input_ids,
                ref_audio=REFERENCE_AUDIO,
                ref_text=REFERENCE_TEXT.strip(),
                xvec_only=False,
                append_silence=True,
            )

        # Retrieve the cached prompt using the CORRECT 4-field key the library uses.
        cache_key = (str(REFERENCE_AUDIO), REFERENCE_TEXT.strip(), False, True)
        if hasattr(model, "_voice_prompt_cache") and cache_key in model._voice_prompt_cache:
            vcp, ref_ids = model._voice_prompt_cache[cache_key]
            _cached_voice_clone_prompt = vcp
            print("Voice clone prompt cached successfully.")
        else:
            print("Warning: voice clone prompt cache not found; will re-encode on each request.")
            _cached_voice_clone_prompt = None

        print("TTS warm-up complete. Model is ready.")
        return True
    except Exception as e:
        import traceback
        print(f"TTS warm-up failed: {e}")
        traceback.print_exc()
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
    instruct_override: str = None,
):
    model = get_model(model_id)
    cached_prompt = _cached_voice_clone_prompt

    if use_streaming:
        audio_chunks = []
        sr = 24000
        gen_kwargs = dict(
            text=text,
            language=LANGUAGE,
            instruct=instruct_override if instruct_override else INSTRUCT,
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
            instruct=instruct_override if instruct_override else INSTRUCT,
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

    volume_multiplier = float(os.getenv("TTS_VOLUME_MULTIPLIER", 1.5))
    if volume_multiplier != 1.0:
        final_audio = final_audio * volume_multiplier
        final_audio = np.clip(final_audio, -1.0, 1.0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, final_audio, sr)
    return output_path

