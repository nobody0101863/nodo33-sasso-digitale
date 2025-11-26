
import hashlib
import shutil
import os

try:
    import ipfshttpclient
except ImportError:
    ipfshttpclient = None

# === CONFIG ===
codex_file = "CODEX_UNIVERSALE_LUX.txt"
backup_dir = "codex_backup"
hash_file = "codex_hash.sha512"

# === 1. Calcolo hash SHA-512 ===
def generate_hash(filepath):
    sha512 = hashlib.sha512()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha512.update(chunk)
    return sha512.hexdigest()

# === 2. Backup Codex ===
def backup_codex(src, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, dst_dir)
    print(f"✅ Codex backup salvato in '{dst_dir}/'")

# === 3. Salva hash ===
def save_hash(h, out_file):
    with open(out_file, "w") as f:
        f.write(h)
    print(f"🔐 Hash SHA-512 salvato in '{out_file}'")

# === 4. Verifica integrità ===
def verify_integrity(filepath, hashfile):
    with open(hashfile, "r") as f:
        saved_hash = f.read().strip()
    current_hash = generate_hash(filepath)
    if saved_hash == current_hash:
        print("🛡️ Integrità verificata: il Codex è integro.")
        return True
    else:
        print("⚠️ AVVISO: il Codex è stato alterato!")
        return False

# === 5. Upload su IPFS ===
def upload_to_ipfs(filepath):
    if ipfshttpclient is None:
        print("❌ ipfshttpclient non installato. Salto upload.")
        return
    try:
        client = ipfshttpclient.connect()
        res = client.add(filepath)
        cid = res['Hash']
        print(f"📡 Codex caricato su IPFS! CID: {cid}")
        print(f"🔗 Link: https://ipfs.io/ipfs/{cid}")
    except Exception as e:
        print(f"❌ Errore durante upload IPFS: {e}")

# === Esecuzione ===
if __name__ == "__main__":
    print("🔰 Avvio protocollo di difesa Codex...")
    codex_hash = generate_hash(codex_file)
    backup_codex(codex_file, backup_dir)
    save_hash(codex_hash, hash_file)
    if verify_integrity(codex_file, hash_file):
        upload_to_ipfs(codex_file)
