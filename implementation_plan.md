# Implementation Plan: Rick TTS — Full Feature Upgrade

## Goal
Extend the `new_rick_tts_fixed` project with three incremental features:
1. **Audio file output** — save the generated audio to a WAV file.
2. **CLI refactor** — expose `chunk_size`, `temperature`, `top_p`, `save_file`, and `stream_play` as command-line arguments.
3. **ChatGPT-style Web UI** — a beautiful single-page app where the user chats with Gemini, and every AI reply is voiced by Rick Sanchez, with audio pre-generated and served on demand via a Play button.

---

## Feature 1: Audio File Output + Save/Stream Flags

### Overview
Add two new boolean flags to the voice cloning script:
- `--save_file` (default `True`) — after all chunks are collected, concatenate and write a `.wav` file to disk.
- `--stream_play` (default `False`) — play audio live through speakers via `StreamPlayer`.

### Files Affected
- **[MODIFY]** [qwen3_tts_voice_clonning_rick_sanchez.py](file:///d:/Data%20Analyst%20courses/portfolio/rick%20tts/new_rick_tts_fixed/qwen3_tts_voice_clonning_rick_sanchez.py)

### Tasks

- [ ] Import `soundfile`, `numpy`, `argparse`, and `pathlib.Path` at the top.
- [ ] Wrap all existing constants into an `argparse.ArgumentParser` block with the following args:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--ref_audio` | str | `rick_sanchez.mp3` | Path to the reference audio |
| `--output` | str | `output.wav` | Path to save the output WAV |
| `--chunk_size` | int | `8` | TTS streaming chunk size |
| `--temperature` | float | `0.85` | Generation temperature |
| `--top_p` | float | `0.9` | Nucleus sampling parameter |
| `--save_file` | flag (store_true) | `True` | Save audio to disk |
| `--stream_play` | flag (store_true) | `False` | Play audio through speakers |

- [ ] Collect all `audio_chunk` arrays in a list during the streaming loop.
- [ ] After the loop ends, if `--save_file`: `numpy.concatenate` all chunks and write with `soundfile.write(output_path, audio, sr)`.
- [ ] If `--stream_play`: instantiate `StreamPlayer` and feed chunks into it during streaming (as it is now).
- [ ] If neither flag is set: print a warning — *"No output mode selected. Use --save_file or --stream_play."*

### Test Block
```bash
# Test: save to file (default)
.\venv\Scripts\python qwen3_tts_voice_clonning_rick_sanchez.py --save_file

# Test: stream to speakers
.\venv\Scripts\python qwen3_tts_voice_clonning_rick_sanchez.py --stream_play

# Test: both at once
.\venv\Scripts\python qwen3_tts_voice_clonning_rick_sanchez.py --save_file --stream_play --output my_test.wav

# Test: custom params
.\venv\Scripts\python qwen3_tts_voice_clonning_rick_sanchez.py --save_file --chunk_size 12 --temperature 0.9 --top_p 0.95
```

**Expected:** `output.wav` (or specified file) appears in the folder after successful run.

---

## Feature 2: Backend TTS Engine (`tts_engine.py`)

### Overview
Extract the TTS logic from the script into a reusable Python module `tts_engine.py` that can be imported by both the CLI script and the web server.

### Files Affected
- **[NEW]** `new_rick_tts_fixed/tts_engine.py`

### Tasks

- [ ] Create `tts_engine.py` with a `generate_audio(text, output_path, chunk_size, temperature, top_p, stream_play) -> str` function.
- [ ] The function loads the model (or uses a pre-loaded singleton to avoid reloading on every request).
- [ ] It uses `model.generate_voice_clone_streaming(...)` with all the same existing parameters.
- [ ] It accumulates chunks, concatenates with `numpy`, saves to `output_path` using `soundfile.write`, and returns `output_path`.
- [ ] Model singleton pattern: load once at module import time, re-use for all calls.

### Test Block
```python
# In a Python shell inside the venv:
from tts_engine import generate_audio
path = generate_audio("Hello Morty!", output_path="test_out.wav")
print(f"Audio saved to: {path}")
# Verify: test_out.wav exists and is playable
```

---

## Feature 3: Web UI — ChatGPT-Style Rick Sanchez Chat

### Overview
A single-page application with:
- A ChatGPT-style dark-themed chat interface.
- User types a message → hits Send → Gemini API returns a Rick Sanchez-styled response in text.
- **In the background**, the text is immediately sent to the TTS engine, which generates and saves a `.wav` file.
- The AI message bubble shows the text + a **Play ▶** button.
- Clicking Play streams the pre-generated audio to the browser.

### Architecture

```
Browser (HTML/CSS/JS)
    ↕ REST API
Flask Backend (app.py)
    ├── POST /chat       → calls Gemini API → returns text + task_id
    ├── POST /generate   → calls tts_engine.generate_audio() → saves wav → returns filename
    └── GET  /audio/<f>  → serves the .wav file to the browser
```

### Files Affected
- **[NEW]** `new_rick_tts_fixed/app.py` — Flask backend
- **[NEW]** `new_rick_tts_fixed/static/index.html` — UI shell
- **[NEW]** `new_rick_tts_fixed/static/style.css` — ChatGPT-style dark theme
- **[NEW]** `new_rick_tts_fixed/static/app.js` — frontend logic
- **[NEW]** `new_rick_tts_fixed/audio_cache/` — directory where TTS WAV files are stored
- **[MODIFY]** `new_rick_tts_fixed/requirements.txt` — add `flask`, `google-generativeai`

### Tasks

#### Step 3.1 — Flask Backend (`app.py`)

- [ ] Install Flask and the Gemini SDK:
  ```bash
  .\venv\Scripts\pip install flask google-generativeai
  ```
- [ ] Create `app.py` with routes:
  - `GET /` → serve `static/index.html`
  - `POST /chat` body: `{"message": "..."}` → call `gemini.GenerativeModel("gemini-2.0-flash").generate_content(prompt)` → return `{"reply": "...", "audio_id": "<uuid>"}`
  - The `/chat` route **immediately** kicks off a background thread calling `tts_engine.generate_audio(reply, output_path=f"audio_cache/{uuid}.wav")`.
  - `GET /audio/<audio_id>` → return the `.wav` file from `audio_cache/`.
  - `GET /audio_status/<audio_id>` → return `{"ready": true/false}` so the frontend can poll.
- [ ] Read `GEMINI_API_KEY` from an environment variable (`.env` file, loaded by `python-dotenv`).
- [ ] The Rick Sanchez system prompt is injected into every Gemini request.

#### Step 3.2 — UI Design (`index.html`, `style.css`)

- [ ] Dark theme similar to ChatGPT: `#0d0d0d` background, `#1e1e1e` message container, `#10a37f`-style green accent for the user bubbles, off-white for the AI bubbles.
- [ ] Sidebar with logo ("Rick TTS") and conversation history placeholder.
- [ ] Chat window with:
  - Scrollable message list.
  - User message: right-aligned, dark green bubble.
  - AI message: left-aligned, dark grey bubble with an avatar (Rick icon).
  - **Play ▶ button** below each AI message — grayed out (disabled) while audio is being generated, activated once `/audio_status/<id>` returns `ready: true`.
  - Typing indicator animation while Gemini is responding.
- [ ] Input bar at the bottom (sticky) with a textarea and a send button.
- [ ] Fully responsive layout.

#### Step 3.3 — Frontend Logic (`app.js`)

- [ ] `sendMessage()`: POST to `/chat`, show user bubble, show typing indicator.
- [ ] On response: hide typing indicator, render AI text bubble with a disabled Play button, save `audio_id`.
- [ ] Poll `/audio_status/<audio_id>` every 1 second until `ready: true`, then enable the Play button.
- [ ] Play button click: `new Audio('/audio/<audio_id>').play()` — streams directly from the Flask server.
- [ ] Handle keyboard shortcut: `Shift+Enter` = newline, `Enter` = send.
- [ ] Auto-scroll to latest message.

#### Step 3.4 — Environment Config

- [ ] Create `.env` file template:
  ```
  GEMINI_API_KEY=your_key_here
  ```
- [ ] Add `python-dotenv` to `requirements.txt`.
- [ ] Add `.env` to `.gitignore`.

### Test Block

```bash
# 1. Start the server
.\venv\Scripts\python app.py

# 2. Open browser to http://localhost:5000

# 3. Type a message and hit Enter
# Expected: user bubble appears, typing indicator shows, then Rick's text reply appears
#           Play button is disabled, then enables after ~10-30 seconds of TTS generation

# 4. Click Play
# Expected: Rick's voice plays through the browser

# 5. Test API key missing:
# Remove GEMINI_API_KEY from .env, restart server
# Expected: /chat returns a clear 500 error with message "GEMINI_API_KEY not set"
```

---

## Open Questions

> [!IMPORTANT]
> **Gemini Model**: I'll use `gemini-2.0-flash` by default for speed. Should I use `gemini-2.5-pro` for better quality responses?

> [!IMPORTANT]
> **API Key Input**: Should the UI have a settings panel to enter the Gemini API key directly in the browser (stored in `localStorage`), or should it always come from the `.env` file on the server?

> [!IMPORTANT]
> **Audio Cache Cleanup**: WAV files will accumulate in the `audio_cache/` folder. Should they be auto-deleted after being played, or kept for the session?

---

## Verification Plan

### Automated
- Script runs without error with `--save_file` flag.
- `output.wav` is a valid WAV file (can be opened in any media player).
- Flask server starts without error.

### Manual
- Open browser, send a chat message, verify text reply appears.
- Verify Play button enables after TTS generation completes.
- Verify audio plays correctly through browser.
- Verify GPU utilization increases during TTS generation (via `nvidia-smi`).
