/* Hubmicroo Voice Assistant widget — embeddable on any page.
 *
 * Voice in/out uses the browser's built-in Web Speech API (no install, no API
 * keys). It posts the transcript to the self-hosted backend (/api/chat), which
 * returns a grounded answer + product cards. Speech-to-text and text-to-speech
 * can later be swapped for private self-hosted Whisper/MMS without touching the
 * backend contract.
 */
(function () {
  "use strict";

  // Point this at your backend. Defaults to same origin.
  var API_BASE = (window.HUBMICROO_API_BASE || "") + "/api";

  var LANGS = {
    en: { code: "en-US", name: "English", rtl: false, dir: "ltr" },
    ur: { code: "ur-PK", name: "اردو", rtl: true, dir: "rtl" },
    ar: { code: "ar-SA", name: "العربية", rtl: true, dir: "rtl" },
  };
  var GREETING = {
    en: "Hello! Ask me about our products, prices, delivery or anything on Hubmicroo.",
    ur: "السلام علیکم! ہماری پروڈکٹس، قیمتوں یا ڈیلیوری کے بارے میں پوچھیں۔",
    ar: "مرحبًا! اسألني عن منتجاتنا أو الأسعار أو التوصيل في هب مايكرو.",
  };

  var state = { lang: "en", auto: true, listening: false, muted: false,
                lastAnswer: "" };

  // ---- Speech recognition (speech -> text) -----------------------------
  var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  var recog = null, micSupported = !!SR;

  function buildRecognizer() {
    if (!SR) return null;
    var r = new SR();
    r.lang = LANGS[state.lang].code;
    r.interimResults = true;
    r.continuous = false;
    r.onresult = function (e) {
      var txt = "";
      for (var i = 0; i < e.results.length; i++) txt += e.results[i][0].transcript;
      setTranscript(txt);
      if (e.results[e.results.length - 1].isFinal) {
        stopListening();
        sendMessage(txt);
      }
    };
    r.onerror = function (e) {
      stopListening();
      if (e.error === "not-allowed" || e.error === "service-not-allowed") {
        showTextFallback("Microphone blocked — type your question instead.");
      }
    };
    r.onend = function () { if (state.listening) stopListening(); };
    return r;
  }

  // ---- Speech synthesis (text -> speech) -------------------------------
  function speak(text, lang) {
    if (state.muted || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    var u = new SpeechSynthesisUtterance(text);
    u.lang = LANGS[lang].code;
    var v = (window.speechSynthesis.getVoices() || []).find(function (x) {
      return x.lang && x.lang.toLowerCase().indexOf(lang) === 0;
    });
    if (v) u.voice = v;
    window.speechSynthesis.speak(u);
  }

  // ---- Networking ------------------------------------------------------
  function sendMessage(message) {
    if (!message || !message.trim()) return;
    addBubble(message, "user");
    setTranscript("");          // clear the live voice transcript once sent
    setProcessing(true);
    fetch(API_BASE + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: message, language: state.auto ? null : state.lang }),
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        setProcessing(false);
        if (data.language && LANGS[data.language]) applyLang(data.language, false);
        addBubble(data.answer, "bot");
        renderProducts(data.products || []);
        state.lastAnswer = data.answer;
        speak(data.answer, data.language || state.lang);
      })
      .catch(function () {
        setProcessing(false);
        addBubble("Connection problem. Please try again.", "bot");
      });
  }

  // ---- UI --------------------------------------------------------------
  var el = {};
  function h(tag, cls, html) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  }

  function build() {
    el.fab = h("button", "hm-fab", "🎤");
    el.fab.title = "Voice Assistant";
    el.fab.onclick = openModal;

    el.modal = h("div", "hm-modal hm-hidden");
    el.modal.innerHTML =
      '<div class="hm-card">' +
      '  <div class="hm-head">' +
      '    <span class="hm-title">Hubmicroo Assistant</span>' +
      '    <select class="hm-lang"></select>' +
      '    <button class="hm-mute" title="Mute">🔊</button>' +
      '    <button class="hm-close" title="Close">✕</button>' +
      '  </div>' +
      '  <div class="hm-body"></div>' +
      '  <div class="hm-products"></div>' +
      '  <div class="hm-micwrap">' +
      '    <div class="hm-wave"><i></i><i></i><i></i><i></i><i></i></div>' +
      '    <div class="hm-transcript"></div>' +
      '  </div>' +
      '  <div class="hm-input">' +
      '    <input class="hm-text" placeholder="Type or tap the mic..."/>' +
      '    <button class="hm-mic" title="Speak">🎤</button>' +
      '    <button class="hm-send" title="Send">➤</button>' +
      '  </div>' +
      "</div>";

    document.body.appendChild(el.fab);
    document.body.appendChild(el.modal);

    el.card = el.modal.querySelector(".hm-card");
    el.body = el.modal.querySelector(".hm-body");
    el.products = el.modal.querySelector(".hm-products");
    el.wave = el.modal.querySelector(".hm-wave");
    el.transcript = el.modal.querySelector(".hm-transcript");
    el.textInput = el.modal.querySelector(".hm-text");
    el.langSel = el.modal.querySelector(".hm-lang");

    // Language selector: Auto + the 3 languages.
    var optAuto = h("option", null, "Auto");
    optAuto.value = "auto";
    el.langSel.appendChild(optAuto);
    Object.keys(LANGS).forEach(function (k) {
      var o = h("option", null, LANGS[k].name);
      o.value = k;
      el.langSel.appendChild(o);
    });
    el.langSel.onchange = function () {
      if (this.value === "auto") { state.auto = true; }
      else { state.auto = false; applyLang(this.value, true); }
    };

    el.modal.querySelector(".hm-close").onclick = closeModal;
    el.modal.querySelector(".hm-mic").onclick = toggleListening;
    el.modal.querySelector(".hm-send").onclick = function () {
      var t = el.textInput.value.trim();
      if (t) { el.textInput.value = ""; sendMessage(t); }
    };
    el.textInput.addEventListener("keydown", function (e) {
      if (e.key === "Enter") { var t = this.value.trim(); if (t) { this.value = ""; sendMessage(t); } }
    });
    el.modal.querySelector(".hm-mute").onclick = function () {
      state.muted = !state.muted;
      this.textContent = state.muted ? "🔇" : "🔊";
      if (state.muted && window.speechSynthesis) window.speechSynthesis.cancel();
    };
    if (!micSupported) el.modal.querySelector(".hm-mic").style.display = "none";
  }

  function openModal() {
    el.modal.classList.remove("hm-hidden");
    if (!el.body.children.length) {
      addBubble(GREETING[state.lang], "bot");
      speak(GREETING[state.lang], state.lang);
    }
    el.textInput.focus();
  }
  function closeModal() {
    el.modal.classList.add("hm-hidden");
    stopListening();
    if (window.speechSynthesis) window.speechSynthesis.cancel();
  }

  function applyLang(lang, announce) {
    state.lang = lang;
    el.card.dir = LANGS[lang].dir;
    el.card.classList.toggle("hm-rtl", LANGS[lang].rtl);
    if (recog) recog.lang = LANGS[lang].code;
    if (announce) { addBubble(GREETING[lang], "bot"); speak(GREETING[lang], lang); }
  }

  function toggleListening() { state.listening ? stopListening() : startListening(); }
  function startListening() {
    if (!micSupported) return;
    recog = buildRecognizer();
    try { recog.start(); } catch (e) { return; }
    state.listening = true;
    el.wave.classList.add("hm-active");
    setTranscript("");
  }
  function stopListening() {
    state.listening = false;
    el.wave.classList.remove("hm-active");
    if (recog) { try { recog.stop(); } catch (e) {} }
  }

  function setTranscript(t) { el.transcript.textContent = t; }
  function setProcessing(on) {
    if (on) addBubble('<span class="hm-typing">●●●</span>', "bot", true);
    else { var t = el.body.querySelector(".hm-temp"); if (t) t.remove(); }
  }

  function addBubble(text, who, temp) {
    var b = h("div", "hm-bubble hm-" + who + (temp ? " hm-temp" : ""), text);
    el.body.appendChild(b);
    el.body.scrollTop = el.body.scrollHeight;
  }

  function renderProducts(products) {
    el.products.innerHTML = "";
    products.forEach(function (p) {
      var card = h("a", "hm-pcard");
      card.href = p.url || "#";
      card.target = "_blank";
      var stock = p.in_stock
        ? '<span class="hm-in">In stock</span>'
        : '<span class="hm-out">Out of stock</span>';
      card.innerHTML =
        (p.image ? '<img src="' + p.image + '" alt=""/>' : "") +
        '<div class="hm-pname">' + (p.name || "") + "</div>" +
        '<div class="hm-pprice">' + (p.price != null ? p.price + " " + (p.currency || "") : "") + "</div>" +
        stock;
      el.products.appendChild(card);
    });
  }

  function showTextFallback(msg) { addBubble(msg, "bot"); el.textInput.focus(); }

  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", build);
  else build();
})();
