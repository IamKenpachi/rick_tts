document.addEventListener('DOMContentLoaded', () => {
    // UI-02: Poll TTS status until warm-up is complete
    async function checkTtsStatus() {
        try {
            const res = await fetch('/tts_status');
            const data = await res.json();
            const indicator = document.getElementById('tts-status-indicator');
            const text = document.getElementById('tts-status-text');
            if (data.ready) {
                indicator.className = 'status-chip status-ready';
                text.textContent = 'TTS: Ready';
                // Stop polling once ready
            } else if (!data.available) {
                indicator.className = 'status-chip status-warming';
                text.textContent = 'TTS: Unavailable';
            } else {
                // Still warming up — poll again in 5 seconds
                setTimeout(checkTtsStatus, 5000);
            }
        } catch (e) {
            setTimeout(checkTtsStatus, 10000); // retry on network error
        }
    }
    checkTtsStatus(); // kick off immediately on page load

    const modelSelect = document.getElementById("model-select");

    async function syncModelDropdown() {
        try {
            const res = await fetch("/current_model");
            const data = await res.json();
            if (data.model_id) {
                if (data.model_id.includes("0.6B")) modelSelect.value = "0.6B";
                else if (data.model_id.includes("1.7B")) modelSelect.value = "1.7B";
            }
        } catch (e) {
            console.warn("Could not fetch current model:", e);
        }
    }
    syncModelDropdown();

    modelSelect.addEventListener("change", async function() {
        const selectedSize = this.value;
        modelSelect.disabled = true;
        const indicator = document.getElementById("tts-status-indicator");
        const statusText = document.getElementById("tts-status-text");
        indicator.className = "status-chip status-warming";
        statusText.textContent = "TTS: Switching to " + selectedSize + "...";
        try {
            const res = await fetch("/switch_model", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model_size: selectedSize })
            });
            if (!res.ok) {
                const err = await res.json();
                alert("Failed to switch model: " + err.error);
                modelSelect.disabled = false;
                return;
            }
            setTimeout(() => {
                modelSelect.disabled = false;
                checkTtsStatus();
            }, 2000);
        } catch (e) {
            alert("Network error: " + e.message);
            modelSelect.disabled = false;
        }
    });

    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    const chatContainer = document.getElementById('chat-container');

    // Auto-resize textarea
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.value.trim() !== '') {
            sendBtn.disabled = false;
        } else {
            sendBtn.disabled = true;
        }
    });

    // Enter to send, Shift+Enter for newline
    messageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', sendMessage);

    function appendUserMessage(text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message user-message';
        msgDiv.innerHTML = `
            <div class="avatar user-avatar">👤</div>
            <div class="message-content">
                <div class="text">${escapeHTML(text)}</div>
            </div>
        `;
        chatContainer.appendChild(msgDiv);
        scrollToBottom();
    }

    function appendAIMessageLoading() {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message ai-message';
        msgDiv.id = 'loading-message';
        msgDiv.innerHTML = `
            <div class="avatar ai-avatar">⚗️</div>
            <div class="message-content">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        chatContainer.appendChild(msgDiv);
        scrollToBottom();
        return msgDiv;
    }

    async function sendMessage() {
        const text = messageInput.value.trim();
        if (!text) return;

        messageInput.value = '';
        messageInput.style.height = 'auto';
        sendBtn.disabled = true;

        appendUserMessage(text);
        const loadingDiv = appendAIMessageLoading();

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: text })
            });

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Server error');
            }

            // Replace loading with actual response
            loadingDiv.removeAttribute('id');
            loadingDiv.querySelector('.message-content').innerHTML = `
                <div class="text">${escapeHTML(data.reply).replace(/\n/g, '<br>')}</div>
                <button class="play-btn" id="play-${data.audio_id}" disabled>
                    <i class="fas fa-spinner fa-spin"></i> Generating Voice...
                </button>
            `;
            
            // Start polling for audio (NEW-08: starts at 2000ms)
            pollAudioStatus(data.audio_id, 2000, 0);

        } catch (error) {
            loadingDiv.removeAttribute('id');
            loadingDiv.querySelector('.message-content').innerHTML = `
                <div class="text" style="color: #ff6b6b;">Error: ${error.message}</div>
            `;
        } finally {
            scrollToBottom();
        }
    }

    const MAX_POLL_ATTEMPTS = 200;

    async function pollAudioStatus(audioId, interval = 2000, attempts = 0) {
        const btn = document.getElementById("play-" + audioId);
        if (!btn) return;

        if (attempts >= MAX_POLL_ATTEMPTS) {
            btn.disabled = false;
            btn.innerHTML = "<i class='fas fa-exclamation-triangle'></i> Timed Out";
            btn.title = "TTS generation took too long. The server may be overloaded.";
            return;
        }

        try {
            const response = await fetch("/audio_status/" + audioId);
            const data = await response.json();

            if (data.ready) {
                btn.disabled = false;
                btn.innerHTML = "<i class='fas fa-play'></i> Play Voice";
                btn._audio = null;
                btn.onclick = () => {
                    if (btn._audio) {
                        btn._audio.pause();
                        btn._audio.currentTime = 0;
                        btn._audio = null;
                    }
                    const audio = new Audio("/audio/" + audioId);
                    btn._audio = audio;
                    btn.disabled = true;
                    btn.classList.add("playing");
                    btn.innerHTML = "<i class='fas fa-volume-up'></i> Playing...";
                    audio.play().catch(err => {
                        console.error("Audio play error:", err);
                        btn.disabled = false;
                        btn.classList.remove("playing");
                        btn.innerHTML = "<i class='fas fa-play'></i> Play Voice";
                        btn._audio = null;
                    });
                    audio.onended = () => {
                        btn.disabled = false;
                        btn.classList.remove("playing");
                        btn.innerHTML = "<i class='fas fa-play'></i> Play Voice";
                        btn._audio = null;
                    };
                };
            } else if (data.failed) {
                btn.disabled = false;
                btn.innerHTML = "<i class='fas fa-exclamation-triangle'></i> Voice Failed";
                btn.title = "TTS Error: " + (data.error || "Unknown error");
            } else {
                if (interval >= 5000 && btn.innerHTML.includes("Generating Voice")) {
                    btn.innerHTML = "<i class='fas fa-clock'></i> Queued...";
                }
                const nextInterval = Math.min(interval * 1.5, 8000);
                setTimeout(() => pollAudioStatus(audioId, nextInterval, attempts + 1), interval);
            }
        } catch (error) {
            console.error("Polling error:", error);
            btn.disabled = false;
            btn.innerHTML = "<i class='fas fa-exclamation-triangle'></i> Retry Audio";
            btn.onclick = () => {
                btn.disabled = true;
                btn.innerHTML = "<i class='fas fa-spinner fa-spin'></i> Retrying...";
                btn.onclick = null;
                pollAudioStatus(audioId, 2000, 0);
            };
        }
    }

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function escapeHTML(str) {
        return str.replace(/[&<>'"]/g, 
            tag => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                "'": '&#39;',
                '"': '&quot;'
            }[tag] || tag)
        );
    }
});
