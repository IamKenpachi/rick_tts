document.addEventListener('DOMContentLoaded', () => {
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    const chatContainer = document.getElementById('chat-container');

    // Auto-resize textarea
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.value.trim() !== '') {
            sendBtn.style.color = 'white';
        } else {
            sendBtn.style.color = '#8e8ea0';
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
            <div class="avatar"><i class="fas fa-user"></i></div>
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
            <div class="avatar"><i class="fas fa-flask"></i></div>
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
            
            // Start polling for audio
            pollAudioStatus(data.audio_id);

        } catch (error) {
            loadingDiv.removeAttribute('id');
            loadingDiv.querySelector('.message-content').innerHTML = `
                <div class="text" style="color: #ff6b6b;">Error: ${error.message}</div>
            `;
        } finally {
            sendBtn.disabled = false;
            scrollToBottom();
        }
    }

    async function pollAudioStatus(audioId) {
        const btn = document.getElementById(`play-${audioId}`);
        if (!btn) return;

        try {
            const response = await fetch(`/audio_status/${audioId}`);
            const data = await response.json();

            if (data.ready) {
                btn.disabled = false;
                btn.innerHTML = `<i class="fas fa-play"></i> Play Voice`;
                btn.onclick = () => {
                    const audio = new Audio(`/audio/${audioId}`);
                    audio.play();
                };
            } else {
                setTimeout(() => pollAudioStatus(audioId), 1000);
            }
        } catch (error) {
            console.error("Polling error:", error);
            btn.innerHTML = `<i class="fas fa-exclamation-triangle"></i> Failed to load audio`;
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
