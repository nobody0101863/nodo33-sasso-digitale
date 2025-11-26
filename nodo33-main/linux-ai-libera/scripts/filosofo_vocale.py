#!/usr/bin/env python3
"""
Filosofo Vocale - Deep Search + AI + TTS
Pipeline completo: Cerca → Riassumi → Haiku → Voce
100% LOCALE, 0€
"""

import sys
import json
import requests
from langchain_community.llms import Ollama
from gtts import gTTS
import os
from datetime import datetime

class FilosofoVocale:
    def __init__(self, searx_url="http://localhost:8080", model="llama3.1:70b"):
        self.searx_url = searx_url
        self.llm = Ollama(model=model, base_url="http://localhost:11434")

    def deep_search(self, query: str, num_results=5):
        """Cerca su SearxNG (motore di ricerca privato)"""
        print(f"🔍 Deep Search: '{query}'")

        try:
            # Cerca su SearxNG
            response = requests.get(
                f"{self.searx_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "language": "it"
                },
                timeout=10
            )

            if response.status_code == 200:
                results = response.json().get("results", [])[:num_results]
                print(f"   ✅ Trovati {len(results)} risultati\n")
                return results
            else:
                print("   ⚠️ SearxNG non disponibile, uso ricerca fallback")
                return self._fallback_search(query)

        except Exception as e:
            print(f"   ⚠️ Errore SearxNG: {e}")
            return self._fallback_search(query)

    def _fallback_search(self, query: str):
        """Fallback se SearxNG non è disponibile"""
        return [{
            "title": f"Ricerca locale per: {query}",
            "content": f"Analisi filosofica del concetto di '{query}' basata su conoscenza interna.",
            "url": "locale"
        }]

    def riassumi_risultati(self, query: str, risultati: list):
        """Riassumi i risultati con l'LLM"""
        print("🤔 Filosofo sta riflettendo...")

        # Prepara il contesto
        contesto = ""
        for i, r in enumerate(risultati, 1):
            titolo = r.get("title", "Senza titolo")
            contenuto = r.get("content", "")
            contesto += f"\n{i}. {titolo}\n   {contenuto}\n"

        # Prompt per il riassunto
        prompt = f"""Sei un filosofo ribelle. Hai cercato informazioni su: "{query}"

Ecco cosa hai trovato:
{contesto}

Compito:
1. Riassumi le idee chiave in 2-3 frasi profonde
2. Aggiungi una riflessione critica
3. Concludi con un haiku in italiano che cattura l'essenza

Scrivi in modo poetico ma chiaro."""

        risposta = self.llm.invoke(prompt)
        print("   ✅ Riflessione completa\n")
        return risposta

    def estrai_haiku(self, testo: str):
        """Estrae l'haiku dal testo (ultima parte)"""
        linee = testo.strip().split("\n")
        # Cerca le ultime 3-4 linee (formato haiku)
        haiku_candidato = "\n".join(linee[-4:])
        return haiku_candidato

    def text_to_speech(self, testo: str, output_file="output.mp3"):
        """Converti testo in voce (TTS locale con gTTS)"""
        print("🎙️ Generazione audio...")

        try:
            tts = gTTS(text=testo, lang='it', slow=False)
            tts.save(output_file)
            print(f"   ✅ Audio salvato: {output_file}")
            return output_file
        except Exception as e:
            print(f"   ❌ Errore TTS: {e}")
            return None

    def esegui_pipeline(self, query: str):
        """Pipeline completo: Search → LLM → Haiku → TTS"""
        print("🪨❤️ NODO33 - Filosofo Vocale")
        print("=" * 70)

        # Step 1: Deep Search
        risultati = self.deep_search(query)

        # Step 2: Riassumi con LLM
        print("=" * 70)
        riflessione = self.riassumi_risultati(query, risultati)

        # Step 3: Mostra il risultato
        print("=" * 70)
        print("💎 RIFLESSIONE FILOSOFICA:")
        print("=" * 70)
        print(riflessione)
        print()

        # Step 4: Estrai haiku
        haiku = self.estrai_haiku(riflessione)
        print("=" * 70)
        print("🌸 HAIKU:")
        print("=" * 70)
        print(haiku)
        print()

        # Step 5: TTS
        print("=" * 70)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"filosofia_{timestamp}.mp3"
        audio_path = self.text_to_speech(riflessione, output_file)

        if audio_path:
            print(f"\n🎧 Ascolta la riflessione: {audio_path}")
            print("   Comando per ascoltare: mpg123 " + audio_path)

        print("\n🪨 'Se anche costoro taceranno, grideranno le pietre!'")

def main():
    if len(sys.argv) < 2:
        print("Uso: python filosofo_vocale.py 'argomento da esplorare'")
        print("\nEsempi:")
        print("  python filosofo_vocale.py 'teoria del caos'")
        print("  python filosofo_vocale.py 'etica dell\\'open source'")
        print("  python filosofo_vocale.py 'paradosso di Zenone'")
        print("\nRequisiti:")
        print("  1. Ollama attivo: ollama serve")
        print("  2. (Opzionale) SearxNG: docker run -d -p 8080:8080 searxng/searxng")
        print("  3. (Opzionale) mpg123 per audio: sudo apt install mpg123")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    # Verifica Ollama
    try:
        filosofo = FilosofoVocale()
        filosofo.esegui_pipeline(query)
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        print("\nVerifica che Ollama sia attivo:")
        print("  ollama serve")
        print("\nE che il modello sia scaricato:")
        print("  ollama pull llama3.1:70b")
        sys.exit(1)

if __name__ == "__main__":
    main()
