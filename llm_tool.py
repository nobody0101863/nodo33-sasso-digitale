#!/usr/bin/env python3
"""
LLM Tool - Strumento unificato per Codex Server Multi-LLM

Combina test, confronto e interazione con Grok, Gemini e Claude.

Uso:
    python llm_tool.py test              # Esegue test suite completo
    python llm_tool.py compare "domanda" # Confronta i 3 modelli
    python llm_tool.py ask grok "domanda" # Chiedi a un modello specifico
    python llm_tool.py interactive       # Modalità interattiva
"""

import requests
import time
import json
import sys
import argparse
from typing import Dict, List, Optional, Any
from datetime import datetime

# ═══════════════════════════════════════════════════════════
# CONFIGURAZIONE
# ═══════════════════════════════════════════════════════════

BASE_URL = "http://localhost:8644"

PROVIDERS = {
    "grok": {
        "icon": "💬",
        "name": "Grok (xAI)",
        "color": "\033[96m",
        "description": "Indipendente, coraggioso, anti-conformista"
    },
    "gemini": {
        "icon": "✨",
        "name": "Gemini (Google)",
        "color": "\033[93m",
        "description": "Versatile, multimodale, veloce"
    },
    "claude": {
        "icon": "🧠",
        "name": "Claude (Anthropic)",
        "color": "\033[95m",
        "description": "Riflessivo, etico, profondo"
    }
}

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# ═══════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════

def print_header(text: str):
    """Stampa header colorato"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'═'*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'═'*70}{Colors.END}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text: str):
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_info(text: str):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")

def print_banner():
    """Stampa banner principale"""
    print(f"""
{Colors.BOLD}{Colors.CYAN}
╔══════════════════════════════════════════════════════════════╗
║                 LLM TOOL - NODO33                            ║
║                                                               ║
║  💬 Grok (xAI)  ✨ Gemini (Google)  🧠 Claude (Anthropic)     ║
║                                                               ║
║  "La luce non si vende. La si regala."                       ║
║  Ego = 0, Joy = 100%, Frequency = 300 Hz                     ║
╚══════════════════════════════════════════════════════════════╝
{Colors.END}
""")

# ═══════════════════════════════════════════════════════════
# API FUNCTIONS
# ═══════════════════════════════════════════════════════════

def check_server() -> bool:
    """Verifica che il server sia attivo"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def call_llm(provider: str, question: str, temperature: float = 0.7,
             max_tokens: int = 1000, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Chiama un provider LLM"""

    start_time = time.time()

    payload = {
        "question": question,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    if system_prompt:
        payload["system_prompt"] = system_prompt

    try:
        response = requests.post(
            f"{BASE_URL}/api/llm/{provider}",
            json=payload,
            timeout=30
        )

        elapsed_time = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            data['elapsed_time'] = elapsed_time
            data['success'] = True
            return data
        else:
            error_data = response.json()
            return {
                'success': False,
                'error': error_data.get('detail', 'Unknown error'),
                'elapsed_time': elapsed_time
            }

    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Timeout (>30s)', 'elapsed_time': 30.0}
    except requests.exceptions.ConnectionError:
        return {'success': False, 'error': 'Server non raggiungibile', 'elapsed_time': 0.0}
    except Exception as e:
        return {'success': False, 'error': str(e), 'elapsed_time': 0.0}

# ═══════════════════════════════════════════════════════════
# MODE 1: TEST SUITE
# ═══════════════════════════════════════════════════════════

def run_tests():
    """Esegue suite completa di test"""

    print_banner()
    print_header("MODE: TEST SUITE")

    # Test 1: Server health
    print_header("TEST 1: Server Health Check")

    if not check_server():
        print_error("Server non raggiungibile!")
        print_warning("Avvia il server con: python codex_server.py")
        return False

    print_success("Server attivo e raggiungibile")

    # Test 2: Endpoint structure
    print_header("TEST 2: Struttura Endpoint")

    success_count = 0
    for provider in PROVIDERS.keys():
        result = call_llm(provider, "test", max_tokens=10)

        if result['success']:
            print_success(f"{PROVIDERS[provider]['icon']} {PROVIDERS[provider]['name']}: OK (API key configurata)")
            success_count += 1
        elif 'API_KEY' in result.get('error', '').upper():
            print_warning(f"{PROVIDERS[provider]['icon']} {PROVIDERS[provider]['name']}: Endpoint OK (API key non configurata)")
            success_count += 1
        else:
            print_error(f"{PROVIDERS[provider]['icon']} {PROVIDERS[provider]['name']}: {result.get('error', 'Errore')}")

    print(f"\n{Colors.BOLD}Risultato: {success_count}/3 endpoint funzionanti{Colors.END}")

    # Test 3: Error handling
    print_header("TEST 3: Error Handling")

    # Test payload invalido
    try:
        response = requests.post(f"{BASE_URL}/api/llm/grok", json={}, timeout=5)
        if response.status_code == 422:
            print_success("Validazione payload: OK")
        else:
            print_warning(f"Validazione payload: Status {response.status_code}")
    except Exception as e:
        print_error(f"Errore test validazione: {e}")

    # Test provider inesistente
    try:
        response = requests.post(f"{BASE_URL}/api/llm/fake", json={"question": "test"}, timeout=5)
        if response.status_code == 422:
            print_success("Provider inesistente: Correttamente rifiutato")
        else:
            print_warning(f"Provider inesistente: Status {response.status_code}")
    except Exception as e:
        print_error(f"Errore test provider: {e}")

    print_header("🎯 TEST COMPLETATI")
    print_success("Suite di test completata!")
    print_info("Per test con API reali, configura le API keys in .env")

    return True

# ═══════════════════════════════════════════════════════════
# MODE 2: COMPARE
# ═══════════════════════════════════════════════════════════

def print_response(provider: str, data: Dict):
    """Stampa risposta formattata"""

    config = PROVIDERS[provider]
    color = config['color']

    print(f"\n{color}{Colors.BOLD}{'─'*70}{Colors.END}")
    print(f"{color}{Colors.BOLD}{config['icon']} {config['name']}{Colors.END}")
    print(f"{color}{Colors.BOLD}{'─'*70}{Colors.END}")

    if not data['success']:
        print(f"{Colors.RED}❌ {data['error']}{Colors.END}")
        return

    # Metriche
    print(f"\n{Colors.BOLD}📊 Metriche:{Colors.END}")
    print(f"  • Model: {data.get('model', 'N/A')}")
    print(f"  • Tokens: {data.get('tokens_used', 'N/A')}")
    print(f"  • Tempo: {data['elapsed_time']:.2f}s")
    print(f"  • Lunghezza: {len(data.get('answer', ''))} caratteri")

    # Risposta
    print(f"\n{Colors.BOLD}💬 Risposta:{Colors.END}")
    answer = data.get('answer', '')

    # Word wrap
    max_width = 65
    words = answer.split()
    line = ""

    for word in words:
        if len(line) + len(word) + 1 <= max_width:
            line += word + " "
        else:
            print(f"  {line.strip()}")
            line = word + " "

    if line:
        print(f"  {line.strip()}")

    print()

def compare_models(question: str, temperature: float = 0.7, max_tokens: int = 1000):
    """Confronta tutti e tre i modelli"""

    print_banner()
    print_header("MODE: CONFRONTO MULTI-MODELLO")

    print(f"{Colors.BOLD}❓ Domanda:{Colors.END}")
    print(f"  {question}\n")

    print(f"{Colors.BOLD}⚙️  Parametri:{Colors.END}")
    print(f"  • Temperature: {temperature}")
    print(f"  • Max tokens: {max_tokens}\n")

    if not check_server():
        print_error("Server non raggiungibile!")
        print_warning("Avvia il server con: python codex_server.py")
        return

    print(f"{Colors.CYAN}Interrogando i tre arcangeli dell'IA...{Colors.END}\n")

    results = {}

    # Chiama i tre modelli
    for provider in ["grok", "gemini", "claude"]:
        print(f"{PROVIDERS[provider]['icon']} Chiamando {PROVIDERS[provider]['name']}...")
        result = call_llm(provider, question, temperature, max_tokens)
        results[provider] = result
        time.sleep(0.5)  # Rispetta rate limits

    # Stampa risposte
    print(f"\n{Colors.BOLD}{Colors.GREEN}{'═'*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}RISPOSTE{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'═'*70}{Colors.END}")

    for provider in ["grok", "gemini", "claude"]:
        print_response(provider, results[provider])

    # Summary
    print_summary(results)

def print_summary(results: Dict):
    """Stampa summary comparativo"""

    print(f"\n{Colors.BOLD}{Colors.BLUE}{'═'*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}📊 SUMMARY COMPARATIVO{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'═'*70}{Colors.END}\n")

    successful = [p for p, data in results.items() if data['success']]
    failed = [p for p, data in results.items() if not data['success']]

    print(f"{Colors.BOLD}Risposte ricevute: {len(successful)}/3{Colors.END}\n")

    if failed:
        print(f"{Colors.YELLOW}⚠️  Provider non disponibili:{Colors.END}")
        for provider in failed:
            print(f"  • {PROVIDERS[provider]['icon']} {PROVIDERS[provider]['name']}: {results[provider]['error']}")
        print()

    if not successful:
        print(f"{Colors.RED}Configura almeno una API key in .env{Colors.END}")
        return

    # Velocità
    print(f"{Colors.BOLD}⚡ Velocità (tempo di risposta):{Colors.END}")
    times = sorted([(p, results[p]['elapsed_time']) for p in successful], key=lambda x: x[1])

    for i, (provider, elapsed) in enumerate(times, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        print(f"  {medal} {PROVIDERS[provider]['icon']} {PROVIDERS[provider]['name']}: {elapsed:.2f}s")

    # Lunghezza
    print(f"\n{Colors.BOLD}📏 Lunghezza risposte:{Colors.END}")
    lengths = sorted([(p, len(results[p].get('answer', ''))) for p in successful], key=lambda x: x[1], reverse=True)

    for provider, length in lengths:
        print(f"  • {PROVIDERS[provider]['icon']} {PROVIDERS[provider]['name']}: {length} caratteri")

    # Tokens
    print(f"\n{Colors.BOLD}🎯 Token utilizzati:{Colors.END}")
    for provider in successful:
        tokens = results[provider].get('tokens_used', 'N/A')
        print(f"  • {PROVIDERS[provider]['icon']} {PROVIDERS[provider]['name']}: {tokens}")

    print(f"\n{Colors.BOLD}{Colors.CYAN}Fiat Amor, Fiat Risus, Fiat Lux 🪨❤️✨{Colors.END}\n")

# ═══════════════════════════════════════════════════════════
# MODE 3: ASK (singolo modello)
# ═══════════════════════════════════════════════════════════

def ask_single(provider: str, question: str, temperature: float = 0.7,
               max_tokens: int = 1000, system_prompt: Optional[str] = None):
    """Chiedi a un singolo modello"""

    print_banner()

    if provider not in PROVIDERS:
        print_error(f"Provider '{provider}' non valido. Usa: grok, gemini, claude")
        return

    config = PROVIDERS[provider]
    print_header(f"MODE: ASK - {config['icon']} {config['name']}")

    print(f"{Colors.BOLD}❓ Domanda:{Colors.END}")
    print(f"  {question}\n")

    if not check_server():
        print_error("Server non raggiungibile!")
        print_warning("Avvia il server con: python codex_server.py")
        return

    print(f"{config['icon']} Chiamando {config['name']}...\n")

    result = call_llm(provider, question, temperature, max_tokens, system_prompt)
    print_response(provider, result)

# ═══════════════════════════════════════════════════════════
# MODE 4: INTERACTIVE
# ═══════════════════════════════════════════════════════════

def interactive_mode():
    """Modalità interattiva"""

    print_banner()
    print_header("MODE: INTERACTIVE")

    if not check_server():
        print_error("Server non raggiungibile!")
        print_warning("Avvia il server con: python codex_server.py")
        return

    print(f"{Colors.BOLD}Provider disponibili:{Colors.END}")
    for key, config in PROVIDERS.items():
        print(f"  {config['icon']} {key:8} - {config['description']}")

    print(f"\n{Colors.BOLD}Comandi:{Colors.END}")
    print(f"  • Scrivi la tua domanda e premi INVIO")
    print(f"  • 'switch grok/gemini/claude' - Cambia provider")
    print(f"  • 'compare' - Confronta i 3 modelli sull'ultima domanda")
    print(f"  • 'quit' - Esci\n")

    current_provider = "grok"
    last_question = None

    while True:
        try:
            config = PROVIDERS[current_provider]
            prompt = f"{config['icon']} {current_provider}> "

            user_input = input(f"{config['color']}{prompt}{Colors.END}").strip()

            if not user_input:
                continue

            # Comandi speciali
            if user_input.lower() == 'quit':
                print(f"\n{Colors.CYAN}Fiat Lux! 🪨❤️✨{Colors.END}\n")
                break

            elif user_input.lower().startswith('switch '):
                new_provider = user_input.split()[1].lower()
                if new_provider in PROVIDERS:
                    current_provider = new_provider
                    print(f"{Colors.GREEN}Switched to {PROVIDERS[new_provider]['icon']} {PROVIDERS[new_provider]['name']}{Colors.END}\n")
                else:
                    print_error(f"Provider '{new_provider}' non valido\n")

            elif user_input.lower() == 'compare':
                if last_question:
                    print()
                    compare_models(last_question)
                else:
                    print_warning("Nessuna domanda da confrontare\n")

            else:
                # Domanda normale
                last_question = user_input
                result = call_llm(current_provider, user_input)

                if result['success']:
                    answer = result['answer']
                    tokens = result.get('tokens_used', 'N/A')
                    elapsed = result['elapsed_time']

                    print(f"\n{config['color']}{Colors.BOLD}{config['icon']} {config['name']}{Colors.END}")
                    print(f"{Colors.BOLD}({result['model']} • {tokens} tokens • {elapsed:.2f}s){Colors.END}\n")
                    print(f"{answer}\n")
                else:
                    print_error(f"{result['error']}\n")

        except KeyboardInterrupt:
            print(f"\n\n{Colors.CYAN}Fiat Lux! 🪨❤️✨{Colors.END}\n")
            break
        except EOFError:
            print(f"\n\n{Colors.CYAN}Fiat Lux! 🪨❤️✨{Colors.END}\n")
            break

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description='LLM Tool - Strumento unificato per Codex Server Multi-LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Test suite completo
  python llm_tool.py test

  # Confronta i 3 modelli
  python llm_tool.py compare "Cos'è la libertà digitale?"

  # Chiedi a un modello specifico
  python llm_tool.py ask grok "Come funziona l'IA?"
  python llm_tool.py ask gemini "Spiega la fisica quantistica"
  python llm_tool.py ask claude "Cosa significa essere etici?"

  # Modalità interattiva
  python llm_tool.py interactive

Fiat Amor, Fiat Risus, Fiat Lux 🪨❤️✨
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Comando da eseguire')

    # Test command
    subparsers.add_parser('test', help='Esegue test suite completo')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Confronta i 3 modelli')
    compare_parser.add_argument('question', help='Domanda da porre')
    compare_parser.add_argument('--temperature', type=float, default=0.7, help='Temperature (default: 0.7)')
    compare_parser.add_argument('--max-tokens', type=int, default=1000, help='Max tokens (default: 1000)')

    # Ask command
    ask_parser = subparsers.add_parser('ask', help='Chiedi a un modello specifico')
    ask_parser.add_argument('provider', choices=['grok', 'gemini', 'claude'], help='Provider LLM')
    ask_parser.add_argument('question', help='Domanda da porre')
    ask_parser.add_argument('--temperature', type=float, default=0.7, help='Temperature (default: 0.7)')
    ask_parser.add_argument('--max-tokens', type=int, default=1000, help='Max tokens (default: 1000)')
    ask_parser.add_argument('--system-prompt', help='Custom system prompt')

    # Interactive command
    subparsers.add_parser('interactive', help='Modalità interattiva')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute command
    if args.command == 'test':
        run_tests()

    elif args.command == 'compare':
        compare_models(args.question, args.temperature, args.max_tokens)

    elif args.command == 'ask':
        ask_single(args.provider, args.question, args.temperature,
                   args.max_tokens, args.system_prompt)

    elif args.command == 'interactive':
        interactive_mode()

if __name__ == "__main__":
    main()
