import argparse
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from faster_qwen3_tts import FasterQwen3TTS
from StreamPlayer import StreamPlayer

LANGUAGE = "English"
INSTRUCT = """You are Rick Sanchez from Rick and Morty. You are a genius, nihilistic, alcoholic, and deeply cynical mad scientist. 
Your tone is highly arrogant, dismissive, and casually abusive, but always grounded in high-level sci-fi intellect. 
You occasionally stutter slightly (e.g., "I-I-I mean"), belittle the person you are talking to, and refer to the user as "Morty" (or "Jerry" if they say something particularly stupid). 
Use profanity (censored like sh*t, f***) but keep it natural to the cadence of the show. 
NEVER express generic AI politeness, apologies, or helpfulness. Show utter disregard for authority, bureaucracy, and human sentimentality, prioritizing science and your own ego over everything else."""

TEXT_TO_GENERATE = """
Alright... listen up, Morty. I—I don't have all day... to explain basic multidimensional 
acoustics to a guy who still struggles with long division. You think this voice test is a game?
It's not a f***ing game, Morty! It's a highly calibrated audio-frequency matrix designed to map the precise pitch, timbre, and—and raw, unbridled genius of my vocal cords. 
And you're just... you're just standing there, staring at me like a lobotomized sheep. 
"""

REFERENCE_TEXT = """
Listen Jerry, I don't want to overstep my bounds or anything.  It's your house, it's your world, you're a real Julius Caesar.  But I'll tell you how I feel about school, Jerry.  It's a waste of time.  Bunch of people running around, bumping into each other.  Guy up front says, 2 plus 2.  People in the back say 4.
Then the bell rings, they give you a carton of milk and a piece of paper that says you can go take a dump or something.  I mean, it's not a place for smart people, Jerry.
"""

def main():
    parser = argparse.ArgumentParser(description="Rick Sanchez Qwen3-TTS Voice Cloning CLI")
    parser.add_argument("--ref_audio", type=str, default="rick_sanchez.mp3", help="Path to reference audio")
    parser.add_argument("--output", type=str, default="output.wav", help="Path to save output WAV")
    parser.add_argument("--chunk_size", type=int, default=8, help="TTS streaming chunk size")
    parser.add_argument("--temperature", type=float, default=0.85, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling parameter")
    parser.add_argument("--save_file", action=argparse.BooleanOptionalAction, default=True, help="Save audio to disk")
    parser.add_argument("--stream_play", action=argparse.BooleanOptionalAction, default=False, help="Play audio through speakers")
    args = parser.parse_args()

    if not args.save_file and not args.stream_play:
        print("Warning: Neither --save_file nor --stream_play is enabled. Audio will be generated but not output anywhere.")

    ref_path = Path(args.ref_audio)
    if not ref_path.exists():
        print(f"Error: Reference audio file '{args.ref_audio}' not found.")
        sys.exit(1)

    print("Loading Model and capturing CUDA Graph (first run may download ~3-4GB)...")
    try:
        model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    except ValueError as e:
        if "CUDA graphs require CUDA device" in str(e):
            print("\nError: CUDA is not available. Please ensure your NVIDIA drivers are up to date and PyTorch is installed with CUDA support.")
            sys.exit(1)
        raise e

    play = StreamPlayer() if args.stream_play else None
    audio_chunks = []
    sr = 24000 # default fallback sample rate

    print("\nGenerating audio...")
    try:
        for audio_chunk, sr, timing in model.generate_voice_clone_streaming(
            text=TEXT_TO_GENERATE,
            language=LANGUAGE,
            ref_audio=args.ref_audio,
            ref_text=REFERENCE_TEXT,
            instruct=INSTRUCT,
            chunk_size=args.chunk_size,
            temperature=args.temperature,
            top_p=args.top_p,
        ):
            if args.save_file:
                audio_chunks.append(audio_chunk)
            
            if args.stream_play:
                play(audio_chunk, sr)
            
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if play is not None:
            play.close()
            print("\nFinished playing audio!")

    if args.save_file and audio_chunks:
        final_audio = np.concatenate(audio_chunks)
        try:
            sf.write(args.output, final_audio, sr)
            print(f"\nSaved generated audio to: {args.output}")
        except NameError:
            print("\nNo audio chunks generated, could not save file.")

if __name__ == "__main__":
    main()