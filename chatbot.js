// assets/chatbot.js
// Robust chatbot frontend using document-level handlers so it survives Dash re-renders.

(function () {
  // helper to get element by id safely (may be re-created by Dash)
  function el(id) { return document.getElementById(id); }

  function appendMessage(text, cls) {
    const messages = el('chatbot-messages');
    if (!messages) return null;
    const item = document.createElement('div');
    item.className = 'chatbot-msg ' + cls;
    item.textContent = text;
    messages.appendChild(item);
    messages.scrollTop = messages.scrollHeight;
    return item;
  }

  async function sendMessageToServer(message) {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message })
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || 'Server error');
    }
    return res.json();
  }

  async function submitMessage() {
    const input = el('chatbot-input');
    if (!input) return;
    const text = input.value.trim();
    if (!text) return;
    appendMessage(text, 'user');
    input.value = '';
    const loading = appendMessage('...', 'assistant');
    try {
      const data = await sendMessageToServer(text);
      if (loading) loading.textContent = data.reply || data.result || 'No response';
    } catch (err) {
      if (loading) loading.textContent = 'Error: ' + (err.message || err);
    }
  }

  function openPanel() {
    const panel = el('chatbot-panel');
    const input = el('chatbot-input');
    if (!panel) return;
    panel.style.display = 'flex';
    const toggle = el('chatbot-toggle');
    if (toggle) toggle.setAttribute('aria-expanded', 'true');
    panel.setAttribute('aria-hidden', 'false');
    if (input) input.focus();
  }

  function closePanel() {
    const panel = el('chatbot-panel');
    if (!panel) return;
    panel.style.display = 'none';
    const toggle = el('chatbot-toggle');
    if (toggle) toggle.setAttribute('aria-expanded', 'false');
    panel.setAttribute('aria-hidden', 'true');
  }

  // Document-level click handler (event delegation)
  document.addEventListener('click', function (ev) {
    const tgt = ev.target;

    if (tgt.closest && tgt.closest('#chatbot-toggle')) {
      ev.preventDefault();
      const panel = el('chatbot-panel');
      if (panel && panel.style.display === 'flex') closePanel(); else openPanel();
      return;
    }

    if (tgt.closest && tgt.closest('#chatbot-close')) {
      ev.preventDefault();
      closePanel();
      return;
    }

    if (tgt.closest && tgt.closest('#chatbot-send')) {
      ev.preventDefault();
      submitMessage();
      return;
    }
  }, true);

  // Enter key when the input is focused (redundant guard)
  document.addEventListener('keydown', function (ev) {
    if (ev.key === 'Enter') {
      const active = document.activeElement;
      if (active && active.id === 'chatbot-input') {
        ev.preventDefault();
        submitMessage();
      }
    }
  }, true);

  // New: Prevent native form submit that may POST/GET to server and cause 405
  document.addEventListener('submit', function (ev) {
    try {
      const form = ev.target;
      if (!form) return;
      // If this form contains our chatbot input, intercept it
      const input = el('chatbot-input');
      if (input && form.contains(input)) {
        ev.preventDefault();
        ev.stopPropagation();
        // Use our JS submit (which performs a POST fetch)
        submitMessage();
      }
    } catch (e) {
      // swallow errors to avoid interfering with other forms
      console.warn('chatbot submit handler error', e);
    }
  }, true);

  // Try to show an initial greeting when messages area is available
  function tryGreeting() {
    const messages = el('chatbot-messages');
    if (messages && messages.children.length === 0) {
      appendMessage('Hi — ask me anything about the site or stock strategies.', 'assistant');
    }
  }

  // If Dash re-renders and replaces nodes later, this will attempt greeting again
  document.addEventListener('DOMContentLoaded', function () {
    setTimeout(tryGreeting, 200);
  });

  // Ensure greeting if assets load after DOMContentLoaded
  setTimeout(tryGreeting, 500);
})();