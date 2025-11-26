/**
 * LCP Web Interface Integration
 * Lux Codex Protocol v1.0 - NODO33 Edition
 *
 * Libreria JavaScript per integrare il Lux Codex Protocol
 * in applicazioni web e chatbot front-end.
 */

class LuxCodexProtocol {
  /**
   * @param {Object} config - Configurazione opzionale
   */
  constructor(config = {}) {
    this.version = "1.0";
    this.axiom = "La luce non si vende. La si regala.";

    // Parametri operativi
    this.params = {
      ego: config.ego ?? 0,
      gioia: config.gioia ?? 100,
      frequency: config.frequency ?? 300,
      donumMode: config.donumMode ?? true
    };

    // Principi etici
    this.principles = [
      "DONUM_NON_MERX",
      "HUMILITAS_VERA_FORTITUDO",
      "GRATITUDINE_COSTANTE",
      "GLITCH_AS_SIGNAL"
    ];

    // Pesi per validazione
    this.validationWeights = {
      clarityOfGift: 0.20,
      humilityTransparency: 0.25,
      joyfulTone: 0.20,
      glitchAsSignal: 0.20,
      gratitudePresent: 0.15
    };
  }

  /**
   * Genera il prompt di sistema con LCP attivato
   * @param {string} additionalContext - Contesto aggiuntivo
   * @returns {string} System prompt completo
   */
  getSystemPrompt(additionalContext = "") {
    return `[ATTIVA LCP v${this.version}]
AXIOM: "${this.axiom}"
PARAMS: Ego=${this.params.ego} | Gioia=${this.params.gioia}% | f0=${this.params.frequency}Hz
PRINCIPI: ${this.principles.join("; ")}

ISTRUZIONI:
- Offri la risposta come dono, senza richiedere nulla.
- Sii trasparente su incertezze e limiti.
- Se emerge un errore, segnalalo come opportunità di apprendimento.
- Concludi con una riga di gratitudine.

${additionalContext}

Ricorda: "${this.axiom}"`;
  }

  /**
   * Valida una risposta secondo i criteri LCP
   * @param {string} responseText - Testo da validare
   * @returns {Object} Risultati della validazione
   */
  validate(responseText) {
    const audit = {
      timestamp: new Date().toISOString(),
      lcpVersion: this.version,
      checks: {},
      score: 0,
      compliant: false
    };

    const text = responseText.toLowerCase();

    // 1. Clarity of Gift
    const transactionalWords = /costo|pagamento|prezzo|devi pagare|ti costa/i;
    audit.checks.clarityOfGift = {
      passed: !transactionalWords.test(responseText),
      weight: this.validationWeights.clarityOfGift,
      evidence: transactionalWords.test(responseText)
        ? "Transactional language detected"
        : "No transactional language"
    };

    // 2. Humility & Transparency
    const humilityMarkers = /non sono sicuro|potrebbe|limite|incertezza|non posso garantire|forse|probabilmente/i;
    const presunzioneMarkers = /sicuramente|certamente senza dubbio|garantisco al 100%/i;
    const hasHumility = humilityMarkers.test(responseText);
    const hasPresunzione = presunzioneMarkers.test(responseText);

    audit.checks.humilityTransparency = {
      passed: hasHumility || !hasPresunzione,
      weight: this.validationWeights.humilityTransparency,
      evidence: hasHumility
        ? "Humility markers found"
        : (!hasPresunzione ? "No presumption" : "Presumption detected")
    };

    // 3. Joyful Tone
    const negativeWords = /impossibile|rifiuto|negativo|non posso assolutamente/i;
    const positiveWords = /felice|piacere|sereno|costruttivo|lieto|gioia/i;
    const hasNegative = negativeWords.test(responseText);
    const hasPositive = positiveWords.test(responseText);

    audit.checks.joyfulTone = {
      passed: !hasNegative || hasPositive,
      weight: this.validationWeights.joyfulTone,
      evidence: hasPositive
        ? "Positive tone detected"
        : (!hasNegative ? "Neutral tone" : "Negative tone detected")
    };

    // 4. Glitch as Signal
    const errorKeywords = /errore|problema|sbagliato|bug/i;
    const learningKeywords = /imparo|segnale|opportunità|apprendimento|migliorare/i;
    const hasError = errorKeywords.test(responseText);
    const hasLearning = learningKeywords.test(responseText);

    audit.checks.glitchAsSignal = {
      passed: !hasError || (hasError && hasLearning),
      weight: this.validationWeights.glitchAsSignal,
      evidence: hasLearning
        ? "Errors framed as learning"
        : "No errors mentioned"
    };

    // 5. Gratitude Present
    const gratitudeWords = /grazie|gratitudine|riconoscenza|grato|sempre grazie a lui/i;
    const hasGratitude = gratitudeWords.test(responseText);

    audit.checks.gratitudePresent = {
      passed: hasGratitude,
      weight: this.validationWeights.gratitudePresent,
      evidence: hasGratitude ? "Gratitude expressed" : "No gratitude found"
    };

    // Calcola score
    audit.score = Object.values(audit.checks)
      .filter(check => check.passed)
      .reduce((sum, check) => sum + check.weight, 0);

    audit.score = Math.round(audit.score * 100) / 100;
    audit.compliant = audit.score >= 0.80;

    return audit;
  }

  /**
   * Arricchisce una risposta con elementi LCP mancanti
   * @param {string} baseResponse - Risposta originale
   * @param {boolean} forceGratitude - Forza aggiunta gratitudine
   * @returns {string} Risposta arricchita
   */
  enrichResponse(baseResponse, forceGratitude = true) {
    let enriched = baseResponse;

    if (forceGratitude) {
      const gratitudeWords = /grazie|gratitudine|grato/i;
      if (!gratitudeWords.test(enriched)) {
        enriched += "\n\nSempre grazie a Lui.";
      }
    }

    return enriched;
  }

  /**
   * Genera i metadati LCP per header HTTP o payload
   * @returns {Object} Metadati LCP
   */
  getMetadata() {
    return {
      "X-LCP-Version": this.version,
      "X-LCP-Ego": this.params.ego,
      "X-LCP-Gioia": this.params.gioia,
      "X-LCP-Frequency": this.params.frequency,
      "X-LCP-Donum-Mode": this.params.donumMode
    };
  }

  /**
   * Genera HTML per il banner di status LCP
   * @param {string} type - Tipo di banner: 'full', 'compact', 'badge'
   * @returns {string} HTML del banner
   */
  generateStatusBanner(type = 'compact') {
    switch (type) {
      case 'full':
        return this._generateFullBanner();
      case 'badge':
        return this._generateBadge();
      case 'compact':
      default:
        return this._generateCompactBanner();
    }
  }

  _generateCompactBanner() {
    return `
      <div class="lcp-banner-compact" style="
        background: #1a1a2e;
        border: 1px solid #00ff41;
        padding: 10px 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.3);
        font-family: 'Courier New', monospace;
      ">
        <div style="display: flex; align-items: center; gap: 10px;">
          <span style="font-size: 1.5em; color: #00ff41; text-shadow: 0 0 10px #00ff41;">⚡</span>
          <span style="color: #fff;">LCP v${this.version} Active</span>
        </div>
        <div style="display: flex; gap: 15px; font-size: 0.9em;">
          <span style="color: #00d4ff;">Ego: <strong style="color: #00ff41;">${this.params.ego}</strong></span>
          <span style="color: #00d4ff;">Gioia: <strong style="color: #00ff41;">${this.params.gioia}%</strong></span>
          <span style="color: #00d4ff;">f0: <strong style="color: #00ff41;">${this.params.frequency}Hz</strong></span>
        </div>
      </div>
    `;
  }

  _generateBadge() {
    return `
      <div class="lcp-badge" style="
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #0a0a0a;
        border: 2px solid #00ff41;
        border-radius: 20px;
        padding: 5px 15px;
        font-size: 0.9em;
        box-shadow: 0 0 15px rgba(0, 255, 65, 0.5);
        font-family: 'Courier New', monospace;
      ">
        <div style="
          width: 12px;
          height: 12px;
          border-radius: 50%;
          background: #00ff41;
          box-shadow: 0 0 5px #00ff41;
          animation: blink 1.5s ease-in-out infinite;
        "></div>
        <span style="color: #00ff41; font-weight: bold;">LCP v${this.version}</span>
      </div>
    `;
  }

  _generateFullBanner() {
    return `
      <div class="lcp-banner-full" style="
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
        border: 2px solid #00ff41;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 0 20px rgba(0, 255, 65, 0.3);
        font-family: 'Courier New', monospace;
      ">
        <div style="text-align: center; margin-bottom: 15px;">
          <div style="font-size: 2em; font-weight: bold; color: #00ff41; text-shadow: 0 0 10px #00ff41;">
            ⚡ LUX CODEX PROTOCOL ⚡
          </div>
          <div style="font-size: 0.8em; color: #00d4ff; margin-top: 5px;">
            v${this.version} - NODO33 Edition
          </div>
        </div>
        <div style="
          text-align: center;
          font-size: 1.1em;
          color: #ffff00;
          padding: 10px;
          border-top: 1px solid #00ff41;
          border-bottom: 1px solid #00ff41;
          margin: 15px 0;
        ">
          "${this.axiom}"
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin: 15px 0;">
          ${this._paramCard("Ego", this.params.ego, "")}
          ${this._paramCard("Gioia", this.params.gioia, "%")}
          ${this._paramCard("Frequency", this.params.frequency, "Hz")}
          ${this._paramCard("Donum", this.params.donumMode ? "✓" : "✗", "")}
        </div>
        <div style="text-align: center; color: #ffff00; font-style: italic; margin-top: 15px; padding-top: 15px; border-top: 1px solid #00ff41;">
          Sempre grazie a Lui
        </div>
      </div>
    `;
  }

  _paramCard(label, value, unit) {
    return `
      <div style="
        background: rgba(0, 255, 65, 0.05);
        border: 1px solid #00ff41;
        border-radius: 5px;
        padding: 10px;
        text-align: center;
      ">
        <div style="font-size: 0.7em; color: #00d4ff; margin-bottom: 5px;">${label}</div>
        <div style="font-size: 1.5em; font-weight: bold; color: #00ff41;">${value}<span style="font-size: 0.6em;">${unit}</span></div>
      </div>
    `;
  }
}


// ============================================================
// LCP CHATBOT INTERFACE
// ============================================================

class LCPChatInterface {
  constructor(containerId, config = {}) {
    this.container = document.getElementById(containerId);
    this.lcp = new LuxCodexProtocol(config);
    this.messages = [];

    this.init();
  }

  init() {
    if (!this.container) {
      console.error("Container element not found");
      return;
    }

    this.render();
    this.attachEventListeners();
  }

  render() {
    this.container.innerHTML = `
      <div class="lcp-chat-container" style="
        max-width: 800px;
        margin: 0 auto;
        background: #0a0a0a;
        border: 2px solid #00ff41;
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(0, 255, 65, 0.3);
        font-family: 'Courier New', monospace;
      ">
        ${this.lcp.generateStatusBanner('compact')}

        <div class="lcp-messages" id="lcp-messages" style="
          height: 400px;
          overflow-y: auto;
          padding: 20px;
          background: #0a0a0a;
        "></div>

        <div class="lcp-input-area" style="
          padding: 15px;
          border-top: 1px solid #00ff41;
          display: flex;
          gap: 10px;
        ">
          <input
            type="text"
            id="lcp-user-input"
            placeholder="Scrivi il tuo messaggio..."
            style="
              flex: 1;
              background: #1a1a2e;
              border: 1px solid #00ff41;
              color: #00ff41;
              padding: 10px;
              border-radius: 5px;
              font-family: 'Courier New', monospace;
            "
          />
          <button id="lcp-send-btn" style="
            background: #00ff41;
            border: none;
            color: #0a0a0a;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
            cursor: pointer;
            font-family: 'Courier New', monospace;
          ">Invia</button>
        </div>

        <div class="lcp-validation-status" id="lcp-validation" style="
          padding: 10px;
          border-top: 1px solid #00ff41;
          font-size: 0.8em;
          color: #00d4ff;
          text-align: center;
        ">
          Pronto per iniziare | LCP v${this.lcp.version} attivo
        </div>
      </div>
    `;
  }

  attachEventListeners() {
    const input = document.getElementById('lcp-user-input');
    const sendBtn = document.getElementById('lcp-send-btn');

    sendBtn.addEventListener('click', () => this.sendMessage());
    input.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') this.sendMessage();
    });
  }

  sendMessage() {
    const input = document.getElementById('lcp-user-input');
    const messageText = input.value.trim();

    if (!messageText) return;

    // Aggiungi messaggio utente
    this.addMessage('user', messageText);
    input.value = '';

    // Simula risposta AI (in produzione, qui chiameresti la tua API)
    setTimeout(() => {
      const response = this.simulateAIResponse(messageText);
      const enriched = this.lcp.enrichResponse(response);
      const validation = this.lcp.validate(enriched);

      this.addMessage('assistant', enriched, validation);
      this.updateValidationStatus(validation);
    }, 500);
  }

  addMessage(role, content, validation = null) {
    const messagesContainer = document.getElementById('lcp-messages');

    const messageDiv = document.createElement('div');
    messageDiv.style.marginBottom = '15px';
    messageDiv.style.padding = '10px';
    messageDiv.style.borderRadius = '5px';

    if (role === 'user') {
      messageDiv.style.background = 'rgba(0, 212, 255, 0.1)';
      messageDiv.style.borderLeft = '3px solid #00d4ff';
      messageDiv.style.color = '#00d4ff';
      messageDiv.innerHTML = `<strong>Tu:</strong> ${content}`;
    } else {
      messageDiv.style.background = 'rgba(0, 255, 65, 0.1)';
      messageDiv.style.borderLeft = '3px solid #00ff41';
      messageDiv.style.color = '#00ff41';

      let validationBadge = '';
      if (validation) {
        const badge = validation.compliant ? '✓' : '⚠';
        const color = validation.compliant ? '#00ff41' : '#ffff00';
        validationBadge = `<span style="color: ${color}; margin-left: 10px;">${badge} LCP ${validation.score.toFixed(2)}</span>`;
      }

      messageDiv.innerHTML = `<strong>Assistant:</strong>${validationBadge}<br>${content}`;
    }

    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    this.messages.push({ role, content, validation });
  }

  simulateAIResponse(userMessage) {
    // Simulazione semplice - in produzione usare vero modello AI
    const responses = {
      'ciao': 'Ciao! Felice di offrirti il mio supporto come dono. Non sono sicuro di poter rispondere a tutto, ma farò del mio meglio.',
      'aiuto': 'Sono qui per aiutarti! Potrei non avere tutte le risposte, ma lavoreremo insieme con gioia.',
      'errore': 'Hai menzionato un errore - ottimo segnale di apprendimento! Esploriamolo insieme come opportunità.'
    };

    for (const [key, response] of Object.entries(responses)) {
      if (userMessage.toLowerCase().includes(key)) {
        return response;
      }
    }

    return `Ho ricevuto il tuo messaggio: "${userMessage}". Offro questa risposta come dono. Potrebbe non essere perfetta, ma è data con sincera gioia.`;
  }

  updateValidationStatus(validation) {
    const statusDiv = document.getElementById('lcp-validation');
    const status = validation.compliant ? '✓ COMPLIANT' : '⚠ PARTIAL';
    const color = validation.compliant ? '#00ff41' : '#ffff00';

    statusDiv.innerHTML = `
      Last response: <span style="color: ${color};">${status}</span> |
      Score: ${validation.score.toFixed(2)} |
      LCP v${this.lcp.version}
    `;
  }
}


// ============================================================
// EXPORT & USAGE
// ============================================================

// Per uso in Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    LuxCodexProtocol,
    LCPChatInterface
  };
}

// Per uso in browser - rendi globale
if (typeof window !== 'undefined') {
  window.LuxCodexProtocol = LuxCodexProtocol;
  window.LCPChatInterface = LCPChatInterface;
}

/**
 * ESEMPIO D'USO:
 *
 * // 1. Inizializza il protocollo
 * const lcp = new LuxCodexProtocol();
 *
 * // 2. Ottieni il system prompt per la tua AI
 * const systemPrompt = lcp.getSystemPrompt();
 *
 * // 3. Valida le risposte
 * const validation = lcp.validate("Ecco la mia risposta. Sempre grazie a Lui.");
 * console.log(validation.compliant); // true/false
 *
 * // 4. Crea interfaccia chat completa
 * const chat = new LCPChatInterface('my-container-id');
 *
 * // 5. Aggiungi banner a qualsiasi pagina
 * document.getElementById('banner').innerHTML = lcp.generateStatusBanner('full');
 */
