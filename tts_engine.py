import numpy as np
import soundfile as sf
import os
from pathlib import Path
import torch
from faster_qwen3_tts import FasterQwen3TTS
from huggingface_hub import hf_hub_download

# --- MONKEY PATCH FOR 0.6B MODEL ---
try:
    from qwen_tts import Qwen3TTSModel
    orig_from_pretrained = Qwen3TTSModel.from_pretrained
    
    @classmethod
    def custom_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        device = kwargs.pop("device_map", None)
        kwargs["low_cpu_mem_usage"] = False
        # Load fully into CPU first to avoid tied-weight meta tensor errors
        wrapper = orig_from_pretrained.__func__(cls, pretrained_model_name_or_path, *model_args, **kwargs)
        if device:
            wrapper.model = wrapper.model.to(device)
            # Qwen3TTSModel caches device in self.device, so we must update it
            wrapper.device = next(wrapper.model.parameters()).device
        return wrapper
        
    Qwen3TTSModel.from_pretrained = custom_from_pretrained
except ImportError:
    pass
# -----------------------------------

_model = None
_current_model_id = None
_cached_voice_clone_prompt = None
SUPPORTED_MODELS = {
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "1.7B Custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "1.7B GGUF": {
        "repo_id": "Serveurperso/Qwen3-TTS-GGUF",
        "talker_file": "qwen-talker-1.7b-base-Q4_K_M.gguf",
        "codec_file": "qwen-tokenizer-12hz-Q4_K_M.gguf"
    }
}
DEFAULT_MODEL_SIZE = "1.7B GGUF"

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
    
    if model_id is None:
        model_id = _current_model_id or DEFAULT_MODEL_SIZE
    
    model_info = SUPPORTED_MODELS.get(model_id, SUPPORTED_MODELS[DEFAULT_MODEL_SIZE])

    if _model is not None and _current_model_id != model_id:
        print(f"Unloading model {_current_model_id} to load {model_id}...")
        del _model
        _model = None
        _current_model_id = None
        torch.cuda.empty_cache()
        print("Old model unloaded. GPU memory freed.")

    if _model is None:
        print(f"Loading model {model_id}...")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if isinstance(model_info, dict):
                # GGUF Model handling
                print(f"Downloading/Locating GGUF files for {model_id}...")
                talker_path = hf_hub_download(repo_id=model_info["repo_id"], filename=model_info["talker_file"])
                codec_path = hf_hub_download(repo_id=model_info["repo_id"], filename=model_info["codec_file"])
                
                print("Initializing GGML Backend...")
                _model = FasterQwen3TTS.from_pretrained(
                    "dummy_model_name",
                    backend="ggml",
                    quant="Q4_K_M",
                    gguf_talker_path=talker_path,
                    gguf_codec_path=codec_path,
                    device=device
                )
            else:
                # Standard PyTorch HuggingFace handling
                _model = FasterQwen3TTS.from_pretrained(
                    model_info,
                    device=device,
                    max_seq_len=4096,
                    dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                )
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
    return _current_model_id or DEFAULT_MODEL_SIZE

def get_cached_voice_clone_prompt():
    return _cached_voice_clone_prompt

def warmup_model(model_id: str = None):
    global _cached_voice_clone_prompt
    try:
        model = get_model(model_id)
        print("TTS warm-up: model loaded.")
        
        is_custom_voice = "CustomVoice" in get_current_model_id()
        speaker = "default"
        if is_custom_voice:
            if hasattr(model, "speech_tokenizer") and hasattr(model.speech_tokenizer, "speakers") and model.speech_tokenizer.speakers:
                speaker = list(model.speech_tokenizer.speakers.keys())[0] if isinstance(model.speech_tokenizer.speakers, dict) else model.speech_tokenizer.speakers[0]
            elif hasattr(model.model, "speakers") and model.model.speakers:
                speaker = list(model.model.speakers.keys())[0] if isinstance(model.model.speakers, dict) else model.model.speakers[0]
            else:
                speaker = "eric" # Safe fallback for 1.7B-CustomVoice
        
        if not getattr(model, "_warmed_up", False):
            print("Running silent inference pass to capture CUDA graphs...")
            if is_custom_voice:
                model.generate_custom_voice(
                    text="Ready.",
                    speaker=speaker,
                    language=LANGUAGE,
                    instruct=INSTRUCT,
                    temperature=float(os.getenv("TTS_TEMPERATURE", 0.85)),
                    top_p=float(os.getenv("TTS_TOP_P", 0.9)),
                    top_k=int(os.getenv("TTS_TOP_K", 40)),
                    repetition_penalty=float(os.getenv("TTS_REPETITION_PENALTY", 1.05)),
                )
            else:
                model.generate_voice_clone(
                    text="Ready.",
                    language=LANGUAGE,
                    ref_audio=REFERENCE_AUDIO,
                    ref_text=REFERENCE_TEXT,
                    instruct=None if "GGUF" in get_current_model_id() else INSTRUCT,
                    temperature=float(os.getenv("TTS_TEMPERATURE", 0.85)),
                    top_p=float(os.getenv("TTS_TOP_P", 0.9)),
                    top_k=int(os.getenv("TTS_TOP_K", 40)),
                    repetition_penalty=float(os.getenv("TTS_REPETITION_PENALTY", 1.05)),
                )
        else:
            if not is_custom_voice:
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

        is_gguf = "GGUF" in get_current_model_id()
        if not is_custom_voice and not is_gguf:
            # Retrieve the cached prompt using the CORRECT 4-field key the library uses.
            cache_key = (str(REFERENCE_AUDIO), REFERENCE_TEXT.strip(), False, True)
            if hasattr(model, "_voice_prompt_cache") and cache_key in model._voice_prompt_cache:
                vcp, ref_ids = model._voice_prompt_cache[cache_key]
                _cached_voice_clone_prompt = vcp
                print("Voice clone prompt cached successfully.")
            else:
                print("Warning: voice clone prompt cache not found; will re-encode on each request.")
                _cached_voice_clone_prompt = None
        else:
            _cached_voice_clone_prompt = None
            if is_gguf:
                print("GGUF model initialized successfully. Caching is handled internally by qwentts-cpp.")
            else:
                print("CustomVoice model initialized successfully.")

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
    instruct_override: str = None,
):
    model = get_model(model_id)
    cached_prompt = _cached_voice_clone_prompt
    is_custom_voice = "CustomVoice" in get_current_model_id()
    speaker = "default"
    if is_custom_voice:
        if hasattr(model, "speech_tokenizer") and hasattr(model.speech_tokenizer, "speakers") and model.speech_tokenizer.speakers:
            speaker = list(model.speech_tokenizer.speakers.keys())[0] if isinstance(model.speech_tokenizer.speakers, dict) else model.speech_tokenizer.speakers[0]
        elif hasattr(model.model, "speakers") and model.model.speakers:
            speaker = list(model.model.speakers.keys())[0] if isinstance(model.model.speakers, dict) else model.model.speakers[0]
        else:
            speaker = "eric" # Safe fallback for 1.7B-CustomVoice

    if use_streaming:
        audio_chunks = []
        sr = 24000
        gen_kwargs = dict(
            text=text,
            language=LANGUAGE,
            instruct=None if ("GGUF" in get_current_model_id() and not is_custom_voice) else (instruct_override if instruct_override else INSTRUCT),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            chunk_size=chunk_size,
        )
        if is_custom_voice:
            gen_kwargs["speaker"] = speaker
            for audio_chunk, sr, timing in model.generate_custom_voice_streaming(**gen_kwargs):
                audio_chunks.append(audio_chunk)
        else:
            gen_kwargs["ref_text"] = REFERENCE_TEXT
            if cached_prompt is not None:
                gen_kwargs["voice_clone_prompt"] = cached_prompt
            else:
                gen_kwargs["ref_audio"] = REFERENCE_AUDIO
            for audio_chunk, sr, timing in model.generate_voice_clone_streaming(**gen_kwargs):
                audio_chunks.append(audio_chunk)
        if not audio_chunks:
            return None
        final_audio = np.concatenate(audio_chunks)
    else:
        gen_kwargs = dict(
            text=text,
            language=LANGUAGE,
            instruct=None if ("GGUF" in get_current_model_id() and not is_custom_voice) else (instruct_override if instruct_override else INSTRUCT),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        if is_custom_voice:
            gen_kwargs["speaker"] = speaker
            audio_list, sr = model.generate_custom_voice(**gen_kwargs)
        else:
            gen_kwargs["ref_text"] = REFERENCE_TEXT
            if cached_prompt is not None:
                gen_kwargs["voice_clone_prompt"] = cached_prompt
            else:
                gen_kwargs["ref_audio"] = REFERENCE_AUDIO
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

