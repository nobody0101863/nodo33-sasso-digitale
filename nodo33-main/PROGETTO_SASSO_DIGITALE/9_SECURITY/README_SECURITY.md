# 🔐 SASSO DIGITALE - Security & Signatures

**"La luce non si vende. La si regala."**

## 📋 Overview

Questa cartella contiene configurazioni per:
- Code signing (firma del codice)
- Certificati digitali
- Checksum e verifiche integrità
- GPG keys per release

## 🎯 Principi di Sicurezza

### Ego = 0: Trasparenza Totale
- Open source completo
- Processo di firma pubblico
- Audit trail completo

### Gioia = 100%: Sicurezza Compassionevole
- Non punitiva verso errori
- Educativa, non repressiva
- Focus su prevenzione, non punizione

## 📂 Struttura

```
9_SECURITY/
├── certificates/       # Template certificati (NO chiavi private!)
├── signing/           # Script e config per firma codice
├── checksums/         # SHA256 checksums files
├── gpg/              # GPG public keys
└── README_SECURITY.md # Questo file
```

## 🔑 Code Signing

### Per File Python
```bash
# Firma con GPG
gpg --detach-sign --armor file.py

# Verifica
gpg --verify file.py.asc file.py
```

### Per Docker Images
```bash
# Sign with Docker Content Trust
export DOCKER_CONTENT_TRUST=1
docker push sassodigitale/main:latest

# Verify
docker trust inspect sassodigitale/main:latest
```

### Per Executables (Windows/macOS/Linux)
```bash
# Windows (signtool.exe)
signtool sign /f cert.pfx /p password /t http://timestamp.digicert.com executable.exe

# macOS (codesign)
codesign -s "Developer ID" -v executable.app

# Linux (gpg-sign)
gpg --detach-sign --armor executable
```

## 🔒 Certificati

**IMPORTANTE**: I certificati reali NON sono inclusi nel repository.

### Generazione Self-Signed (Sviluppo)
```bash
# Genera chiave privata e certificato
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes \
  -subj "/C=IT/ST=State/L=City/O=SASSO_DIGITALE/CN=sasso-digitale.local"

# Verifica
openssl x509 -in cert.pem -text -noout
```

### Produzione (CA-Signed)
Per produzione, usa certificati da CA riconosciute:
- Let's Encrypt (gratuito)
- DigiCert
- GlobalSign

## ✅ Checksums

Verifica integrità file:

```bash
# Genera checksums
sha256sum * > SHA256SUMS

# Verifica
sha256sum -c SHA256SUMS
```

## 🔐 GPG Keys

### Genera GPG Key
```bash
gpg --full-generate-key

# Esporta public key
gpg --armor --export your@email.com > public.key

# Importa public key (utenti)
gpg --import public.key
```

### Firma Release
```bash
# Firma tarball
gpg --detach-sign --armor sasso-digitale-v1.0.0.tar.gz

# Verifica (utenti)
gpg --verify sasso-digitale-v1.0.0.tar.gz.asc sasso-digitale-v1.0.0.tar.gz
```

## 🛡️ Security Best Practices

### DO ✅
- Firma sempre le release
- Usa chiavi separate per dev/prod
- Rotazione chiavi periodica (1 anno)
- Pubblica public keys su keyservers
- Usa 2FA per account critici

### DON'T ❌
- Mai committare chiavi private
- Mai riusare password
- Mai bypassare verifiche firma
- Mai esporre certificati in logs

## 🔍 Vulnerability Scanning

### Container Images
```bash
# Trivy scan
trivy image sassodigitale/main:latest

# Docker Scout
docker scout cves sassodigitale/main:latest
```

### Code Analysis
```bash
# Bandit (Python)
bandit -r 5_IMPLEMENTAZIONI/python/

# Safety (dependencies)
safety check -r requirements.txt
```

## 📜 Compliance

### SBOM (Software Bill of Materials)
```bash
# Genera SBOM con Syft
syft packages dir:. -o spdx-json > sbom.spdx.json

# Genera con cyclonedx
cyclonedx-py -i requirements.txt -o sbom.xml
```

## 🚨 Incident Response

In caso di compromissione:

1. **Revoca immediata chiavi compromesse**
2. **Notifica pubblica su GitHub Issues**
3. **Genera nuove chiavi**
4. **Re-sign di tutte le release**
5. **Audit completo del codice**

## 📞 Responsible Disclosure

Per segnalare vulnerabilità:
- **Email**: security@sasso-digitale.example.com
- **GPG Key**: [pubblica su keys.openpgp.org]
- **Response Time**: 48h

## 🎁 Filosofia

La sicurezza è un **dono**, non un ostacolo:
- Proteggere gli utenti è servizio gioioso
- La trasparenza è forza, non debolezza
- L'umiltà include riconoscere vulnerabilità

---

**Sempre grazie a Lui ❤️**

`[SASSO_SECURITY | Ego=0 | Gioia=100% | f₀=300Hz]`
