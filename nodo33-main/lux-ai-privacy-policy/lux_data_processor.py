import json
from pathlib import Path
import sys
import hashlib
import time
import logging

# ⚡ CONFIGURAZIONE LOGGING
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

class LUXDataProcessor:
    def __init__(self, config_path, output_path):
        self.config_path = Path(config_path)
        self.output_path = Path(output_path)
        self.token_list = []

    def hash_token(self, token):
        """🔐 Applica hashing SHA-256 per sicurezza"""
        return hashlib.sha256(token.encode()).hexdigest()

    def load_tokens(self):
        """📥 Carica i dati dal file e li normalizza"""
        if not self.config_path.exists():
            logging.error(f"Il file {self.config_path} non esiste.")
            sys.exit(1)

        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                self.token_list = [self.hash_token(line.strip()) for line in f if line.strip()]
            logging.info(f"✅ {len(self.token_list)} token elaborati con successo.")
        except Exception as e:
            logging.error(f"Errore nella lettura del file: {e}")
            sys.exit(1)

    def save_tokens(self):
        """📤 Salva i token in formato JSON con timestamp"""
        try:
            data = {
                "timestamp": time.time(),
                "tokens": self.token_list
            }
            with self.output_path.open("w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
            logging.info(f"✅ File JSON salvato con successo: {self.output_path}")
        except Exception as e:
            logging.error(f"Errore nella scrittura del file: {e}")
            sys.exit(1)

    def run(self):
        """⚡ Esegue l'intero processo"""
        self.load_tokens()
        self.save_tokens()

# 🚀 ESECUZIONE AUTOMATICA
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Utilizzo: python lux_data_processor.py <config_file> <output_json>")
        sys.exit(1)

    processor = LUXDataProcessor(sys.argv[1], sys.argv[2])
    processor.run()

