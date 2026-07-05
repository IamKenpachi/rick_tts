# Feature Suggestions — Rick TTS Project

A brainstorm of potential features to add to the Rick TTS chat app, ordered roughly by impact and feasibility.

---

## 🎭 Character & AI Behavior

1. **Multi-character switching** — Add a character selector in the sidebar (e.g., Rick, Morty, Beth, Jerry). Each character gets its own reference audio, system prompt, and personality. Switching clears the chat or starts a new session.

2. **Dynamic Rick mood system** — Rick's tone shifts based on conversation context. If the user asks science-related questions, Rick becomes engaged. If the user says something "dumb", Rick becomes more dismissive. This could be implemented as a secondary Gemini prompt that classifies the user message as `[science/personal/dumb/challenge]` and selects a variant system prompt.

3. **Rick insults the AI model name** — When starting the app, Rick makes a sarcastic remark about running on "Gemini" ("Oh great, I'm powered by Google's ad machine. Fantastic."). Small hardcoded flavor text.

4. **Conversation memory** — Maintain a rolling window of the last N messages as context in every Gemini request, so Rick remembers what was said earlier in the conversation.

5. **Rick mood meter UI element** — A small visual indicator (e.g. a flask that fills up or a tiny animation) showing how annoyed/interested Rick is based on the conversation topic.

---

## 🔊 Audio & Voice

6. **Adjustable voice speed/pitch post-processing** — Use `librosa` (already installed) to apply time-stretching or pitch-shifting to the generated WAV file before serving it, allowing the user to make Rick sound more slurred or more frantic.

7. **Rick "burp" injection** — Rick famously burps mid-sentence. Add a post-processing step that randomly inserts a short burp audio clip at natural pause points in the generated speech using `pydub` (already installed).

8. **Audio download button** — Add a download button next to each Play button so the user can save individual Rick replies as MP3/WAV files.

9. **Background ambient sound** — Play a faint "lab ambience" loop (beeps, bubbling, portal hum) in the browser while the chat is active. Toggleable with a button.

10. **Voice streaming to browser (real-time)** — Instead of waiting for the full audio to generate before enabling the Play button, stream the audio chunks directly to the browser via a Server-Sent Events (SSE) or WebSocket connection. Drastically reduces time-to-first-sound.

11. **Multiple reference audio profiles** — Allow the user to upload their own reference audio for Rick's voice (e.g., a different episode clip), so the voice clone quality can be adjusted.

---

## 🖥️ UI / UX

12. **Dark/Light mode toggle** — Even though dark mode is Rick's natural habitat, some users may prefer a lighter theme. Add a simple CSS variable toggle.

13. **Settings panel (sidebar drawer)** — A slide-out panel to configure: Gemini API key, TTS parameters (chunk_size, temperature, top_p), output directory, character selection, and audio mode (stream/save).

14. **Typing animation for Rick's text** — Instead of the full reply appearing at once, stream Gemini's response token-by-token using the Gemini streaming API, so it looks like Rick is typing in real-time. This also lets TTS start generating earlier.

15. **Message regeneration** — A "regenerate" button on each AI reply that asks Gemini to rephrase the answer, then re-generates the audio.

16. **Copy-to-clipboard button** — A small clipboard icon on each AI message to copy the text.

17. **Chat history persistence** — Save conversations to a local JSON file or SQLite database, and display them in the sidebar for the user to revisit.

18. **Mobile-responsive layout** — The current plan targets desktop. Add a proper bottom-sheet input for mobile and collapsible sidebar.

---

## 🛠️ Developer / Technical

19. **REST API with proper OpenAI-compatible schema** — Expose a `/v1/chat/completions`-style endpoint so the Rick TTS server could theoretically be used as a drop-in voice-enabled Gemini client for other tools.

20. **WebSocket real-time communication** — Replace the HTTP polling for audio status with a WebSocket event that pushes `audio_ready` directly to the client when TTS finishes, for a snappier UX.

21. **GPU memory monitoring** — Display a small GPU VRAM usage indicator in the UI (queried from `nvidia-smi` via a backend `/status` endpoint) so the user can see the model working.

22. **Configurable model selection** — Let the user swap between `Qwen3-TTS-12Hz-0.6B-Base` (faster, lighter) and `Qwen3-TTS-12Hz-1.7B-Base` (better quality) from the settings panel.

23. **Docker container** — Package the entire app (Flask + TTS engine) into a Docker container with NVIDIA CUDA runtime support, making the project trivially deployable on any CUDA-enabled cloud VM.

24. **Benchmark mode** — Add a `--benchmark` flag to the CLI that runs the same text 5 times, measures RTF (Real-Time Factor) and TTFA (Time-To-First-Audio), and prints a summary table — useful for portfolio demonstrations.

---

## 📦 Portfolio / Product

25. **Shareable link / HuggingFace Spaces deployment** — Deploy the web UI to HuggingFace Spaces using `Gradio` or a Docker Space, so the project can be shared publicly as a portfolio piece without requiring the viewer to install anything.

26. **Demo video recording mode** — A button that starts recording the browser screen + audio output using the MediaRecorder API, creating a self-contained demo clip.

27. **"Rick explains your code" mode** — A special prompt mode where the user pastes a code snippet and Rick explains it in character ("Oh great, you wrote a bubble sort. In 2026. Really pushing the boundaries, aren't we, Morty.").
