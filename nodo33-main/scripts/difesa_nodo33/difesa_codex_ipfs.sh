
#!/bin/bash

CODEX_FILE="CODEX_UNIVERSALE_LUX.txt"
BACKUP_DIR="codex_backup"
HASH_FILE="codex_hash.sha512"

# === 1. Calcolo hash SHA-512 ===
generate_hash() {
    sha512sum "$1" | awk '{ print $1 }'
}

# === 2. Backup ===
mkdir -p "$BACKUP_DIR"
cp "$CODEX_FILE" "$BACKUP_DIR/"
echo "✅ Backup creato in $BACKUP_DIR/"

# === 3. Calcolo hash e salvataggio ===
HASH=$(generate_hash "$CODEX_FILE")
echo "$HASH" > "$HASH_FILE"
echo "🔐 Hash salvato in $HASH_FILE"

# === 4. Verifica integrità ===
STORED_HASH=$(cat "$HASH_FILE")
CURRENT_HASH=$(generate_hash "$CODEX_FILE")

if [ "$STORED_HASH" == "$CURRENT_HASH" ]; then
    echo "🛡️ Integrità verificata: il Codex è integro."

    # === 5. Upload IPFS ===
    if ! command -v ipfs &> /dev/null; then
        echo "❌ IPFS non è installato. Salto upload."
        exit 1
    fi

    CID=$(ipfs add -Q "$CODEX_FILE")
    echo "📡 Codex caricato su IPFS! CID: $CID"
    echo "🔗 Link: https://ipfs.io/ipfs/$CID"
else
    echo "⚠️ AVVISO: Il Codex è stato alterato!"
fi
