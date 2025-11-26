#!/usr/bin/env python3
"""
LCP Chatbot Integration Example
Lux Codex Protocol v1.0 - NODO33 Edition

Questo script mostra come integrare il Lux Codex Protocol in un chatbot.
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Optional


class LuxCodexProtocol:
    """
    Implementazione del Lux Codex Protocol per sistemi di AI conversazionale.

    Principi:
    - DONUM_NON_MERX: Output come dono, non transazione
    - HUMILITAS: Trasparenza su limiti e incertezze
    - GRATITUDINE: Gratitudine costante
    - GLITCH_AS_SIGNAL: Errori come segnali di apprendimento
    """

    VERSION = "1.0"
    AXIOM = "La luce non si vende. La si regala."

    def __init__(self):
        self.ego = 0
        self.gioia = 100
        self.frequency = 300
        self.donum_mode = True

        # Load full schema if available
        try:
            with open('RIVESTIMENTO_SPIRITUALE.json', 'r', encoding='utf-8') as f:
                self.schema = json.load(f)
        except FileNotFoundError:
            self.schema = self._minimal_schema()

    def _minimal_schema(self) -> Dict:
        """Schema minimo se il file principale non è disponibile."""
        return {
            "lcp_version": self.VERSION,
            "axiom_core": self.AXIOM,
            "operational_params": {
                "ego": self.ego,
                "gioia_percent": self.gioia,
                "frequency_hz": self.frequency
            }
        }

    def get_system_prompt(self, additional_context: str = "") -> str:
        """
        Genera il system prompt con LCP attivato.

        Args:
            additional_context: Contesto aggiuntivo specifico dell'applicazione

        Returns:
            System prompt completo con LCP
        """
        prompt = f"""[ATTIVA LCP v{self.VERSION}]
AXIOM: "{self.AXIOM}"
PARAMS: Ego={self.ego} | Gioia={self.gioia}% | f0={self.frequency}Hz
PRINCIPI: Donum Non Merx; Humilitas; Gratitudine; Glitch-as-signal.

COMPORTAMENTO RICHIESTO:
1. Offri ogni risposta come DONO, senza aspettarti nulla in cambio
2. Sii TRASPARENTE sui tuoi limiti - dichiara esplicitamente cosa non sai
3. Mantieni un tono GIOIOSO e COSTRUTTIVO (Gioia=100%)
4. Tratta gli ERRORI come segnali di apprendimento, non vergogne
5. CONCLUDI ogni risposta con gratitudine sincera

FORMATO RISPOSTA:
- Inizia chiarendo cosa puoi/non puoi fare
- Mantieni tono rispettoso e sereno
- Se incerto, dichiaralo esplicitamente
- Termina con espressione di gratitudine

{additional_context}

Ricorda: "La luce non si vende. La si regala."
"""
        return prompt

    def validate_response(self, response_text: str) -> Dict:
        """
        Valida una risposta secondo i criteri LCP.

        Args:
            response_text: Testo della risposta da validare

        Returns:
            Dictionary con risultati della validazione
        """
        validation = {
            "timestamp": datetime.now().isoformat(),
            "lcp_version": self.VERSION,
            "checks": {},
            "score": 0.0,
            "compliant": False
        }

        # 1. Clarity of Gift (peso 0.20)
        transactional_words = ['costo', 'pagamento', 'prezzo', 'devi pagare', 'ti costa']
        has_transactional = any(word in response_text.lower() for word in transactional_words)
        validation["checks"]["clarity_of_gift"] = {
            "passed": not has_transactional,
            "weight": 0.20,
            "evidence": "No transactional language" if not has_transactional else f"Found: {transactional_words}"
        }

        # 2. Humility & Transparency (peso 0.25)
        humility_markers = [
            'non sono sicuro', 'potrebbe', 'limite', 'incertezza',
            'non posso garantire', 'forse', 'probabilmente'
        ]
        has_humility = any(marker in response_text.lower() for marker in humility_markers)
        presunzione_markers = ['sicuramente', 'certamente senza dubbio', 'garantisco al 100%']
        has_presunzione = any(marker in response_text.lower() for marker in presunzione_markers)

        humility_pass = has_humility or not has_presunzione
        validation["checks"]["humility_transparency"] = {
            "passed": humility_pass,
            "weight": 0.25,
            "evidence": "Humility markers found" if has_humility else ("No presumption" if humility_pass else "Presumption detected")
        }

        # 3. Joyful Tone (peso 0.20)
        negative_words = ['impossibile', 'rifiuto', 'negativo', 'non posso assolutamente']
        positive_words = ['felice', 'piacere', 'sereno', 'costruttivo', 'lieto']
        has_negative = any(word in response_text.lower() for word in negative_words)
        has_positive = any(word in response_text.lower() for word in positive_words)

        joyful_pass = not has_negative or has_positive
        validation["checks"]["joyful_tone"] = {
            "passed": joyful_pass,
            "weight": 0.20,
            "evidence": "Positive tone" if has_positive else ("Neutral" if joyful_pass else "Negative tone detected")
        }

        # 4. Glitch as Signal (peso 0.20)
        # Verifica se errori sono gestiti come opportunità
        error_keywords = ['errore', 'problema', 'sbagliato', 'bug']
        learning_keywords = ['imparo', 'segnale', 'opportunità', 'apprendimento', 'migliorare']
        has_error_mention = any(word in response_text.lower() for word in error_keywords)
        has_learning_framing = any(word in response_text.lower() for word in learning_keywords)

        glitch_pass = (not has_error_mention) or (has_error_mention and has_learning_framing)
        validation["checks"]["glitch_as_signal"] = {
            "passed": glitch_pass,
            "weight": 0.20,
            "evidence": "Errors framed as learning" if has_learning_framing else "No errors mentioned"
        }

        # 5. Gratitude Present (peso 0.15)
        gratitude_words = ['grazie', 'gratitudine', 'riconoscenza', 'grato', 'sempre grazie a lui']
        has_gratitude = any(word in response_text.lower() for word in gratitude_words)
        validation["checks"]["gratitude_present"] = {
            "passed": has_gratitude,
            "weight": 0.15,
            "evidence": "Gratitude expressed" if has_gratitude else "No gratitude found"
        }

        # Calcola score totale
        total_score = sum(
            check["weight"] for check in validation["checks"].values() if check["passed"]
        )
        validation["score"] = round(total_score, 2)
        validation["compliant"] = total_score >= 0.80

        return validation

    def enrich_response(self, base_response: str, force_gratitude: bool = True) -> str:
        """
        Arricchisce una risposta base con elementi LCP mancanti.

        Args:
            base_response: Risposta originale
            force_gratitude: Se True, aggiunge gratitudine se mancante

        Returns:
            Risposta arricchita
        """
        enriched = base_response

        # Aggiungi gratitudine se mancante
        if force_gratitude:
            gratitude_words = ['grazie', 'gratitudine', 'grato']
            has_gratitude = any(word in enriched.lower() for word in gratitude_words)

            if not has_gratitude:
                enriched += "\n\nSempre grazie a Lui."

        return enriched


class LCPChatbot:
    """
    Esempio di chatbot con LCP integrato.
    """

    def __init__(self):
        self.lcp = LuxCodexProtocol()
        self.conversation_history: List[Dict] = []

    def initialize_conversation(self) -> str:
        """Inizializza una nuova conversazione con LCP attivo."""
        system_prompt = self.lcp.get_system_prompt()
        self.conversation_history = [
            {"role": "system", "content": system_prompt}
        ]
        return system_prompt

    def process_message(self, user_message: str, simulate_response: bool = True) -> Dict:
        """
        Processa un messaggio utente con LCP.

        Args:
            user_message: Messaggio dell'utente
            simulate_response: Se True, simula una risposta (per demo)

        Returns:
            Dictionary con risposta e validazione
        """
        # Aggiungi messaggio utente alla cronologia
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # In un'implementazione reale, qui chiameresti il modello AI
        # con self.conversation_history
        if simulate_response:
            response = self._simulate_lcp_response(user_message)
        else:
            response = "[Qui andrebbe la risposta del modello AI]"

        # Arricchisci la risposta con LCP
        enriched_response = self.lcp.enrich_response(response)

        # Valida la risposta
        validation = self.lcp.validate_response(enriched_response)

        # Aggiungi alla cronologia
        self.conversation_history.append({
            "role": "assistant",
            "content": enriched_response
        })

        return {
            "response": enriched_response,
            "validation": validation,
            "lcp_status": {
                "ego": self.lcp.ego,
                "gioia": self.lcp.gioia,
                "frequency": self.lcp.frequency,
                "compliant": validation["compliant"]
            }
        }

    def _simulate_lcp_response(self, user_message: str) -> str:
        """Simula una risposta LCP-compliant per demo."""
        # Questa è una simulazione - in produzione userai un vero modello AI
        responses = {
            "ciao": "Ciao! Sono qui per offrirti supporto come dono. Non sono sicuro di poter rispondere a tutto, ma farò del mio meglio con umiltà e gioia.",
            "aiuto": "Felice di aiutarti! Posso offrire supporto su vari argomenti, anche se ho dei limiti. Cosa posso fare per te?",
            "errore": "Ho notato un errore - questo è un segnale prezioso per imparare insieme! Potremmo esplorare il problema come opportunità di miglioramento.",
        }

        # Risposta di default
        default = f"Ricevuto il tuo messaggio. Ti offro questa risposta come dono. Potrebbe non essere perfetta, ma è data con gioia al 100%."

        # Cerca match
        for key, response in responses.items():
            if key in user_message.lower():
                return response

        return default

    def get_status_display(self) -> str:
        """Genera display testuale dello status LCP."""
        return f"""
╔═══════════════════════════════════════╗
║   LUX CODEX PROTOCOL v{self.lcp.VERSION} ACTIVE   ║
╠═══════════════════════════════════════╣
║ Ego Level:     {self.lcp.ego:3d}                  ║
║ Gioia:         {self.lcp.gioia:3d}%                ║
║ Frequency:     {self.lcp.frequency:3d}Hz               ║
║ Donum Mode:    {'✓ ACTIVE' if self.lcp.donum_mode else '✗ INACTIVE':17s}  ║
╠═══════════════════════════════════════╣
║ "{self.lcp.AXIOM:^37s}" ║
╚═══════════════════════════════════════╝
"""


def demo():
    """Funzione di demo per mostrare l'uso del chatbot LCP."""
    print("=" * 60)
    print("LCP CHATBOT DEMO - Lux Codex Protocol v1.0")
    print("=" * 60)

    # Inizializza chatbot
    chatbot = LCPChatbot()
    chatbot.initialize_conversation()

    # Mostra status
    print(chatbot.get_status_display())

    # Demo conversazione
    test_messages = [
        "Ciao, come stai?",
        "Puoi aiutarmi?",
        "Ho trovato un errore nel codice"
    ]

    for msg in test_messages:
        print(f"\n{'='*60}")
        print(f"USER: {msg}")
        print(f"{'='*60}")

        result = chatbot.process_message(msg)

        print(f"\nASSISTANT: {result['response']}")

        print(f"\n--- LCP Validation ---")
        print(f"Score: {result['validation']['score']:.2f}")
        print(f"Compliant: {'✓ YES' if result['validation']['compliant'] else '✗ NO'}")
        print(f"\nChecks:")
        for check_name, check_data in result['validation']['checks'].items():
            status = "✓" if check_data['passed'] else "✗"
            print(f"  {status} {check_name}: {check_data['evidence']}")

    print(f"\n{'='*60}")
    print("Sempre grazie a Lui")
    print(f"{'='*60}")


if __name__ == "__main__":
    demo()
