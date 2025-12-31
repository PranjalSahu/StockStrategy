// Wait for DOM to be ready
window.addEventListener('DOMContentLoaded', function() {
    console.log('Chatbot script loaded');
    
    // Wait a bit for Dash to finish rendering
    setTimeout(() => {
        const toggle = document.getElementById('chatbot-toggle');
        const panel = document.getElementById('chatbot-panel');
        const close = document.getElementById('chatbot-close');
        const form = document.getElementById('chatbot-form');
        const input = document.getElementById('chatbot-input');
        const messages = document.getElementById('chatbot-messages');
        const sendBtn = document.getElementById('chatbot-send');

        if (!toggle || !panel || !close || !form || !input || !messages || !sendBtn) {
            console.error('Chatbot elements not found:', {
                toggle: !!toggle,
                panel: !!panel,
                close: !!close,
                form: !!form,
                input: !!input,
                messages: !!messages,
                sendBtn: !!sendBtn
            });
            return;
        }

        console.log('All chatbot elements found');

        // Toggle chatbot panel
        toggle.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('Toggle clicked');
            if (panel.style.display === 'flex') {
                panel.style.display = 'none';
            } else {
                panel.style.display = 'flex';
            }
        });

        // Close chatbot panel
        close.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('Close clicked');
            panel.style.display = 'none';
        });

        // Handle send button click
        const sendMessage = async () => {
            console.log('Send message triggered');
            
            const message = input.value.trim();
            if (!message) return;

            // Add user message
            const userDiv = document.createElement('div');
            userDiv.className = 'chat-message user';
            userDiv.textContent = message;
            messages.appendChild(userDiv);
            
            input.value = '';
            messages.scrollTop = messages.scrollHeight;

            // Add assistant message container
            const assistantDiv = document.createElement('div');
            assistantDiv.className = 'chat-message assistant';
            assistantDiv.textContent = '';
            messages.appendChild(assistantDiv);

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });

                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\n');

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = line.slice(6);
                            if (data === '[DONE]') break;
                            
                            try {
                                const parsed = JSON.parse(data);
                                if (parsed.chunk) {
                                    assistantDiv.textContent += parsed.chunk;
                                    messages.scrollTop = messages.scrollHeight;
                                } else if (parsed.error) {
                                    assistantDiv.textContent = 'Error: ' + parsed.error;
                                }
                            } catch (e) {
                                // Skip invalid JSON
                                console.error('JSON parse error:', e);
                            }
                        }
                    }
                }
            } catch (error) {
                console.error('Fetch error:', error);
                assistantDiv.textContent = 'Error: ' + error.message;
            }
        };

        // Send on button click
        sendBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            sendMessage();
        });

        // Send on Enter key
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                sendMessage();
            }
        });
    }, 500); // Wait 500ms for Dash to render
});