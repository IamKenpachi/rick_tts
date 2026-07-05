from tts_engine import get_model
import traceback

def test():
    print("Testing 1.7B")
    try:
        model = get_model("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        print("1.7B loaded successfully")
    except Exception as e:
        print("1.7B failed:")
        traceback.print_exc()

    print("\nTesting 0.6B")
    try:
        model2 = get_model("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
        print("0.6B loaded successfully")
    except Exception as e:
        print("0.6B failed:")
        traceback.print_exc()

if __name__ == "__main__":
    test()
