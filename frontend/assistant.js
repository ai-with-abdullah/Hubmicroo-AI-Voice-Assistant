/**
 * Hubmicroo Assistant Widget
 * Self-contained embeddable widget — no browser Web Speech API.
 * Audio recording uses MediaRecorder → sends to server Whisper STT.
 */
(function () {
  "use strict";

  const API_BASE = window.HM_API_BASE || "";

  // ── Build DOM ──────────────────────────────────────────────────────────────
  function buildWidget() {
    // Launcher
    const launcher = el("button", { id: "hm-launcher", "aria-label": "Open assistant" });
    launcher.innerHTML = iconChat();

    // Modal
    const modal = el("div", { id: "hm-modal", role: "dialog", "aria-label": "Shopping assistant" });
    modal.innerHTML = `
      <div id="hm-header">
        <div id="hm-header-avatar">🛍️</div>
        <div>
          <div id="hm-header-title">Hubmicroo Assistant</div>
          <div id="hm-header-sub">Ask me anything about our products</div>
        </div>
        <button id="hm-close" aria-label="Close">${iconX()}</button>
      </div>
      <div id="hm-messages" aria-live="polite"></div>
      <div id="hm-status"></div>
      <div id="hm-input-area">
        <textarea id="hm-text-input" rows="1" placeholder="Type a message…" maxlength="500"></textarea>
        <button class="hm-icon-btn" id="hm-mic-btn" aria-label="Record voice">${iconMic()}</button>
        <button class="hm-icon-btn" id="hm-send-btn" aria-label="Send">${iconSend()}</button>
      </div>`;

    document.body.appendChild(launcher);
    document.body.appendChild(modal);

    // Wire events
    launcher.addEventListener("click", toggleModal);
    modal.querySelector("#hm-close").addEventListener("click", toggleModal);
    modal.querySelector("#hm-send-btn").addEventListener("click", handleSend);
    modal.querySelector("#hm-text-input").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
    });
    modal.querySelector("#hm-mic-btn").addEventListener("click", handleMic);

    // Auto-resize textarea
    const ta = modal.querySelector("#hm-text-input");
    ta.addEventListener("input", () => {
      ta.style.height = "auto";
      ta.style.height = Math.min(ta.scrollHeight, 100) + "px";
    });
  }

  // ── State ──────────────────────────────────────────────────────────────────
  let isOpen = false;
  let isRecording = false;
  let mediaRecorder = null;
  let audioChunks = [];
  let micDenied = false;
  let currentLang = "en";

  function toggleModal() {
    isOpen = !isOpen;
    document.getElementById("hm-modal").classList.toggle("hm-open", isOpen);
    if (isOpen && document.getElementById("hm-messages").children.length === 0) {
      sendGreeting();
    }
    if (isOpen) document.getElementById("hm-text-input").focus();
  }

  // ── Greeting ───────────────────────────────────────────────────────────────
  function sendGreeting() {
    const greetings = {
      en: "Hi! I'm the Hubmicroo shopping assistant. How can I help you today?",
      ur: "Salam! Main Hubmicroo ka shopping assistant hoon. Aap ki kya madad kar sakta hoon?",
      ar: "مرحباً! أنا مساعد تسوق هبمايكرو. كيف يمكنني مساعدتك؟",
    };
    appendBotMessage(greetings[currentLang] || greetings.en, [], currentLang);
  }

  // ── Chat ───────────────────────────────────────────────────────────────────
  async function handleSend() {
    const ta = document.getElementById("hm-text-input");
    const msg = ta.value.trim();
    if (!msg) return;
    ta.value = "";
    ta.style.height = "auto";

    appendUserMessage(msg);
    const typing = appendTyping();
    setStatus("");

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      currentLang = data.lang || "en";
      removeTyping(typing);
      appendBotMessage(data.answer, data.products || [], data.lang);
    } catch (err) {
      removeTyping(typing);
      appendBotMessage("Sorry, something went wrong. Please try again.", [], "en");
    }
  }

  // ── Voice ──────────────────────────────────────────────────────────────────
  async function handleMic() {
    if (micDenied) return;
    if (isRecording) {
      stopRecording();
      return;
    }
    await startRecording();
  }

  async function startRecording() {
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      micDenied = true;
      const btn = document.getElementById("hm-mic-btn");
      btn.classList.add("mic-denied");
      btn.title = "Microphone access denied — use text input";
      setStatus("Mic access denied. Please type your question.");
      return;
    }

    audioChunks = [];
    const mimeType = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "audio/ogg";
    mediaRecorder = new MediaRecorder(stream, { mimeType });
    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      processAudio(mimeType);
    };
    mediaRecorder.start();
    isRecording = true;
    document.getElementById("hm-mic-btn").classList.add("recording");
    setStatus("🎙 Recording… tap to stop");
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
    isRecording = false;
    document.getElementById("hm-mic-btn").classList.remove("recording");
    setStatus("Processing…");
  }

  async function processAudio(mimeType) {
    const blob = new Blob(audioChunks, { type: mimeType });
    if (blob.size < 1000) { setStatus(""); return; }

    const typing = appendTyping();
    const formData = new FormData();
    formData.append("audio", blob, "recording.webm");
    formData.append("language", currentLang);
    formData.append("tts", "true");

    try {
      const res = await fetch(`${API_BASE}/api/voice`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      currentLang = data.lang || "en";

      if (data.transcript) appendUserMessage(`🎙 ${data.transcript}`);
      removeTyping(typing);
      appendBotMessage(data.answer, data.products || [], data.lang);

      if (data.audio_b64) playBase64Audio(data.audio_b64);
    } catch (err) {
      removeTyping(typing);
      appendBotMessage("Voice processing failed. Please type your question.", [], "en");
    }
    setStatus("");
  }

  function playBase64Audio(b64) {
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    const blob = new Blob([arr], { type: "audio/wav" });
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.play().catch(() => {});
    audio.onended = () => URL.revokeObjectURL(url);
  }

  // ── DOM helpers ────────────────────────────────────────────────────────────
  function appendUserMessage(text) {
    const msgs = document.getElementById("hm-messages");
    const div = el("div", { class: "hm-msg hm-user" });
    div.innerHTML = `<div class="hm-bubble">${escHtml(text)}</div>`;
    msgs.appendChild(div);
    scrollToBottom();
  }

  function appendBotMessage(text, products, lang) {
    const msgs = document.getElementById("hm-messages");
    const isRtl = lang === "ur" || lang === "ar";
    const div = el("div", { class: `hm-msg hm-bot${isRtl ? " hm-rtl" : ""}` });

    let html = `<div class="hm-bubble">${escHtml(text)}</div>`;

    if (products && products.length > 0) {
      html += `<div class="hm-products">`;
      for (const p of products) {
        const stock = p.in_stock
          ? `<span class="hm-product-stock">In Stock</span>`
          : `<span class="hm-product-stock out">Out of Stock</span>`;
        const img = p.image_url
          ? `<img class="hm-product-img" src="${escAttr(p.image_url)}" alt="${escAttr(p.name)}" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%2252%22 height=%2252%22><rect fill=%22%23f1f5f9%22 width=%2252%22 height=%2252%22/><text x=%2226%22 y=%2232%22 text-anchor=%22middle%22 font-size=%2220%22>📦</text></svg>'">`
          : `<div class="hm-product-img" style="display:flex;align-items:center;justify-content:center;font-size:22px">📦</div>`;
        html += `
          <div class="hm-product-card">
            ${img}
            <div class="hm-product-info">
              <div class="hm-product-name" title="${escAttr(p.name)}">${escHtml(p.name)}</div>
              <div>
                <span class="hm-product-price">${escHtml(String(p.currency))} ${escHtml(String(p.price))}</span>
                ${stock}
              </div>
            </div>
            ${p.buy_url ? `<a class="hm-buy-btn" href="${escAttr(p.buy_url)}" target="_blank" rel="noopener">Buy</a>` : ""}
          </div>`;
      }
      html += `</div>`;
    }

    div.innerHTML = html;
    msgs.appendChild(div);
    scrollToBottom();
  }

  function appendTyping() {
    const msgs = document.getElementById("hm-messages");
    const div = el("div", { class: "hm-msg hm-bot hm-typing", id: "hm-typing" });
    div.innerHTML = `<div class="hm-bubble"><div class="hm-dot"></div><div class="hm-dot"></div><div class="hm-dot"></div></div>`;
    msgs.appendChild(div);
    scrollToBottom();
    return div;
  }

  function removeTyping(el) {
    if (el && el.parentNode) el.parentNode.removeChild(el);
  }

  function setStatus(msg) {
    const s = document.getElementById("hm-status");
    if (s) s.textContent = msg;
  }

  function scrollToBottom() {
    const msgs = document.getElementById("hm-messages");
    msgs.scrollTop = msgs.scrollHeight;
  }

  function el(tag, attrs = {}) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  }

  function escHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function escAttr(s) { return escHtml(s); }

  // ── Icons (inline SVG) ─────────────────────────────────────────────────────
  function iconChat() {
    return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`;
  }
  function iconX() {
    return `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;
  }
  function iconMic() {
    return `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>`;
  }
  function iconSend() {
    return `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>`;
  }

  // ── Init ───────────────────────────────────────────────────────────────────
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", buildWidget);
  } else {
    buildWidget();
  }
})();
