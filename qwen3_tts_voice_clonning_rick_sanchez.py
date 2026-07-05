from faster_qwen3_tts import FasterQwen3TTS
from StreamPlayer import StreamPlayer

# 1. Define your reference files (Make sure the audio file is uploaded to Colab!)
REFERENCE_AUDIO = "rick_sanchez.mp3" 
REFERENCE_TEXT = """
Listen Jerry, I don't want to overstep my bounds or anything.  It's your house, it's your world, you're a real Julius Caesar.  But I'll tell you how I feel about school, Jerry.  It's a waste of time.  Bunch of people running around, bumping into each other.  Guy up front says, 2 plus 2.  People in the back say 4.
Then the bell rings, they give you a carton of milk and a piece of paper that says you can go take a dump or something.  I mean, it's not a place for smart people, Jerry.
"""

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

print("Loading Model and capturing CUDA Graph...")
model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

play = StreamPlayer()
try:
    for audio_chunk, sr, timing in model.generate_voice_clone_streaming(
        text=TEXT_TO_GENERATE,
        language=LANGUAGE,
        ref_audio=REFERENCE_AUDIO,
        ref_text=REFERENCE_TEXT,
        instruct=INSTRUCT,
        chunk_size=8,  # Yields audio every ~667ms
        temperature=0.85,
        top_p=0.9,
    ):
        # As soon as 667ms of audio is generated, it plays out of your speakers!
        play(audio_chunk, sr)
        
except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    play.close()
    print("\nFinished playing audio!")