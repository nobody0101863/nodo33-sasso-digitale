
import os
import hashlib
import subprocess

# === Codex MilSpec Integration Protocol v1.0 [Python Version] ===

def sha256sum(filename):
    h = hashlib.sha256()
    with open(filename, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def analyze_file(filepath):
    print(f"[+] Analisi file: {filepath}")
    print(f"  SHA256: {sha256sum(filepath)}")

    print("  [*] Header esadecimale:")
    with open(filepath, 'rb') as f:
        hex_sample = f.read(256)
        print('  ', hex_sample.hex(" ", 1))

    print("  [*] Stringhe significative:")
    try:
        strings = subprocess.check_output(['strings', '-n', '6', filepath]).decode()
        for line in strings.splitlines():
            if any(k in line.lower() for k in ['codex', 'proto', 'core', 'entropy', 'harmon', 'sync', 'link']):
                print("   >", line)
    except Exception as e:
        print("  [!] Errore estrazione stringhe:", e)

    print("  [*] Analisi entropia (simulazione):")
    with open(filepath, 'rb') as f:
        data = f.read()
        import math
        from collections import Counter
        freq = Counter(data)
        entropy = -sum(p / len(data) * math.log2(p / len(data)) for p in freq.values())
        print(f"   Entropia stimata: {entropy:.4f} bits/byte")

    print("---")

# File binari da analizzare
files = [
    "codex_integrato.bin",
    "coerenza_dinamica.bin",
    "entropia_armonica.bin",
    "network_framework.bin"
]

for file in files:
    if os.path.exists(file):
        analyze_file(file)
    else:
        print(f"[!] File non trovato: {file}")
