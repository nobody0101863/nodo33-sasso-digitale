import logging

# 📌 Configurazione del logging per salvare i log in un file
logging.basicConfig(
    filename="lux_ai_processor.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
import sqlite3
import json
import time
import logging

class LuxAIProcessor:
    def __init__(self, db_path="lux_tokens.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, isolation_level=None)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        """📌 Crea la tabella se non esiste"""
        self.cursor.execute("CREATE TABLE IF NOT EXISTS tokens (id INTEGER PRIMARY KEY, token TEXT, timestamp REAL);")

    def save_tokens(self, token_list):
        """📤 Salva i token nel database e in JSON"""
        try:
            logging.info(f"🛠 Tentativo di salvataggio di {len(token_list)} token nel database.")

            for token in token_list:
                logging.info(f"🔹 Salvando token nel database: {token}")
                self.cursor.execute("INSERT INTO tokens (token, timestamp) VALUES (?, ?)", (token, time.time()))

            self.conn.commit()
            logging.info("✅ Tutti i token sono stati salvati con successo nel database.")

            self.cursor.execute("SELECT COUNT(*) FROM tokens;")
            count = self.cursor.fetchone()[0]
            logging.info(f"📊 Token attualmente nel database: {count}")

        except sqlite3.Error as e:
            logging.error(f"❌ Errore durante il salvataggio dei token nel database: {e}")

    def export_json(self, filename="output_tokens.json"):
        """📤 Esporta i token in JSON"""
        self.cursor.execute("SELECT token, timestamp FROM tokens;")
        data = [{"token": row[0], "timestamp": row[1]} for row in self.cursor.fetchall()]

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        logging.info(f"✅ Token esportati in {filename}")

# 🚀 ESECUZIONE AUTOMATICA
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    processor = LuxAIProcessor()

    # Simuliamo un input di token
    token_list = ["token1", "token2", "token3"]
    processor.save_tokens(token_list)

    # Esportiamo in JSON
    processor.export_json()
import hashlib
import rsa

class Security:
    """🔐 Blockchain Security: Firma Digitale per i Token"""

    def __init__(self):
        # Genera una coppia di chiavi RSA (privata e pubblica)
        self.public_key, self.private_key = rsa.newkeys(512)

    def sign_token(self, token):
        """✍️ Firma un token con SHA-256 + RSA"""
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        signature = rsa.sign(token_hash.encode(), self.private_key, 'SHA-256')
        return token_hash, signature.hex()

    def verify_signature(self, token, signature):
        """🔍 Verifica la firma di un token"""
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        try:
            rsa.verify(token_hash.encode(), bytes.fromhex(signature), self.public_key)
            return True
        except rsa.VerificationError:
            return False
def save_tokens(self):
    """📤 Salva i token e allena l’AI Evolutiva"""
    try:
        security = Security()
        ai = LuxAIEvolution()

        logging.info(f"🛠 Tentativo di salvataggio di {len(self.token_list)} token nel database.")

        token_lengths = []
        for token in self.token_list:
            token_hash, signature = security.sign_token(token)
            token_lengths.append(len(token))
            self.db.cursor.execute("INSERT INTO tokens (token, timestamp, signature) VALUES (?, ?, ?)",
                                   (token_hash, time.time(), signature))

        self.db.conn.commit()
        ai.train(token_lengths)  # 📡 Allenamento AI
        logging.info("✅ Tutti i token sono stati salvati con successo e firmati.")

    except sqlite3.Error as e:
        logging.error(f"❌ Errore nel salvataggio dei token nel database: {e}")

