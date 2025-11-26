
import ipfshttpclient

# Connessione al nodo locale IPFS
client = ipfshttpclient.connect()

# Percorso del Codex
codex_path = "CODEX_UNIVERSALE_LUX.txt"

# Caricamento su IPFS
res = client.add(codex_path)

# Stampa del CID e link IPFS
cid = res['Hash']
print(f"📡 Codex caricato! CID: {cid}")
print(f"🔗 Link IPFS: https://ipfs.io/ipfs/{cid}")
