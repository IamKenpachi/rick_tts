document.addEventListener('DOMContentLoaded', () => {
    function generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    let currentSessionId = localStorage.getItem('rick_session_id');
    if (!currentSessionId) {
        currentSessionId = generateUUID();
        localStorage.setItem('rick_session_id', currentSessionId);
    }

    // UI-02: Poll TTS status until warm-up is complete
    async function checkTtsStatus() {
        try {
            const res = await fetch('/tts_status');
            const data = await res.json();
            const indicator = document.getElementById('tts-status-indicator');
            const text = document.getElementById('tts-status-text');
            const overlay = document.getElementById("loading-overlay");
            
            if (data.ready) {
                indicator.className = 'status-chip status-ready';
                text.textContent = 'TTS: Ready';
                overlay.style.display = 'none'; // Hide overlay when ready
            } else if (!data.available) {
                indicator.className = 'status-chip status-warming';
                text.textContent = 'TTS: Unavailable';
                overlay.style.display = 'flex'; // Show overlay
            } else {
                // Still warming up
                indicator.className = 'status-chip status-warming';
                text.textContent = 'TTS: Warming up...';
                overlay.style.display = 'flex'; // Show overlay
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
                if (data.model_id.includes("CustomVoice")) modelSelect.value = "1.7B Custom";
                else if (data.model_id.includes("1.7B")) modelSelect.value = "1.7B";
                else if (data.model_id.includes("0.6B")) modelSelect.value = "0.6B";
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
        const overlay = document.getElementById("loading-overlay");
        
        overlay.style.display = "flex"; // Block UI during switch
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
                overlay.style.display = "none";
                return;
            }
            // Model switch is synchronous now, so if it's OK, it's ready!
            setTimeout(() => {
                modelSelect.disabled = false;
                overlay.style.display = "none";
                checkTtsStatus();
            }, 500);
        } catch (e) {
            alert("Network error: " + e.message);
            modelSelect.disabled = false;
            overlay.style.display = "none";
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

        const geminiModel = document.getElementById("gemini-model-select").value;

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: text, session_id: currentSessionId, gemini_model: geminiModel })
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.error || 'Server error');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';
            
            // Replace loading state with empty text
            loadingDiv.removeAttribute('id');
            const contentDiv = loadingDiv.querySelector('.message-content');
            contentDiv.innerHTML = `<div class="text"></div>`;
            const textDiv = contentDiv.querySelector('.text');
            
            let audioId = null;
            let audioInstruction = null;
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                buffer = lines.pop(); // Keep incomplete lines in buffer
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.error) {
                                throw new Error(data.error);
                            }
                            if (data.token) {
                                textDiv.innerHTML += escapeHTML(data.token).replace(/\n/g, '<br>');
                                scrollToBottom();
                            }
                            if (data.done) {
                                audioId = data.audio_id;
                                audioInstruction = data.instruction || "";
                            }
                        } catch (e) {
                            console.error('SSE JSON Error:', e, line);
                        }
                    }
                }
            }

            if (audioId) {
                const btnHtml = `
                <div class="audio-controls-wrapper" style="display: flex; gap: 8px; margin-top: 8px;">
                    <button class="play-btn" id="play-${audioId}" disabled>
                        <i class="fas fa-spinner fa-spin"></i> Generating Voice...
                    </button>
                    <button class="play-btn" id="regen-${audioId}" style="display:none; padding: 6px 12px;" onclick="regenerateAudio('${audioId}', this, '${escapeHTML(audioInstruction).replace(/'/g, "\\'")}')" title="Regenerate Audio">
                        <i class="fas fa-redo"></i>
                    </button>
                </div>`;
                contentDiv.insertAdjacentHTML('beforeend', btnHtml);
                pollAudioStatus(audioId, 2000, 0);
            }

            loadSessions(); // refresh history

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
                const regenBtn = document.getElementById("regen-" + audioId);
                if (regenBtn) regenBtn.style.display = "inline-block";
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

    async function loadSessions() {
        try {
            const res = await fetch('/sessions');
            const sessions = await res.json();
            const list = document.getElementById('session-list');
            if (!list) return;
            list.innerHTML = '';
            sessions.forEach(s => {
                const div = document.createElement('div');
                div.className = 'session-item' + (s.id === currentSessionId ? ' active' : '');
                div.textContent = `Session ${s.id.slice(0,8)}`;
                div.onclick = () => {
                    currentSessionId = s.id;
                    localStorage.setItem('rick_session_id', currentSessionId);
                    loadSessions();
                    loadSessionHistory();
                    if (window.innerWidth <= 768) {
                        document.querySelector('.sidebar').classList.remove('open');
                    }
                };
                list.appendChild(div);
            });
        } catch(e) { console.warn(e); }
    }

    async function loadSessionHistory() {
        chatContainer.innerHTML = '';
        try {
            const res = await fetch('/session/' + currentSessionId);
            const history = await res.json();
            if (history.length === 0) {
                chatContainer.innerHTML = `
                <div class="message ai-message">
                    <div class="avatar ai-avatar">⚗️</div>
                    <div class="message-content">
                        <div class="text">Ugh, great. Another one who figured out how to open a browser. What do you want, Morty? Spit it out — I've got a neutrino bomb that needs calibrating and exactly zero time for your nonsense.</div>
                    </div>
                </div>`;
            } else {
                history.forEach(msg => {
                    if (msg.role === 'user') {
                        appendUserMessage(msg.content);
                    } else {
                        const msgDiv = document.createElement('div');
                        msgDiv.className = 'message ai-message';
                        msgDiv.innerHTML = `
                            <div class="avatar ai-avatar">⚗️</div>
                            <div class="message-content">
                                <div class="text">${escapeHTML(msg.content).replace(/\n/g, '<br>')}</div>
                            </div>
                        `;
                        chatContainer.appendChild(msgDiv);
                    }
                });
            }
            scrollToBottom();
        } catch(e) { console.warn(e); }
    }

    window.startNewSession = () => {
        currentSessionId = generateUUID();
        localStorage.setItem('rick_session_id', currentSessionId);
        loadSessions();
        loadSessionHistory();
        if (window.innerWidth <= 768) {
            document.querySelector('.sidebar').classList.remove('open');
        }
    };

    const sidebarToggle = document.getElementById('sidebar-toggle');
    const sidebar = document.querySelector('.sidebar');
    if (sidebarToggle && sidebar) {
        sidebarToggle.addEventListener('click', () => {
            sidebar.classList.toggle('open');
        });
    }

    window.regenerateAudio = async function(audioId, btnElement, instruction) {
        const wrapper = btnElement.closest('.message-content');
        const textElement = wrapper.querySelector('.text');
        const text = textElement.innerText || textElement.textContent;
        const playBtn = document.getElementById("play-" + audioId);
        
        playBtn.disabled = true;
        playBtn.innerHTML = "<i class='fas fa-spinner fa-spin'></i> Regenerating...";
        btnElement.style.display = "none";
        
        if (playBtn._audio) {
            playBtn._audio.pause();
            playBtn._audio = null;
        }

        try {
            const res = await fetch("/regenerate_audio", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ audio_id: audioId, text: text, instruction: instruction })
            });
            if (!res.ok) throw new Error("Failed to regenerate");
            pollAudioStatus(audioId, 2000, 0);
        } catch (e) {
            playBtn.innerHTML = "<i class='fas fa-exclamation-triangle'></i> Error";
            console.error(e);
        }
    };

    loadSessions();
    loadSessionHistory();
});
