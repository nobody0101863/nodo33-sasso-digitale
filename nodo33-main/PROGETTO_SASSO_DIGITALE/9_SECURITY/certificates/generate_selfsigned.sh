#!/usr/bin/env bash
# ===================================
# SASSO DIGITALE - Self-Signed Certificate Generator
# "La luce non si vende. La si regala."
# For DEVELOPMENT ONLY
# ===================================

set -euo pipefail

CERT_DIR="./certs"
DAYS_VALID=365

echo "🪨 SASSO DIGITALE - Certificate Generator"
echo "Ego=0 | Gioia=100% | f₀=300Hz"
echo

mkdir -p "$CERT_DIR"

# Generate private key and certificate
openssl req -x509 -newkey rsa:4096 \
    -keyout "$CERT_DIR/sasso-key.pem" \
    -out "$CERT_DIR/sasso-cert.pem" \
    -days "$DAYS_VALID" \
    -nodes \
    -subj "/C=IT/ST=Digital/L=Cloud/O=SASSO_DIGITALE/OU=CODEX_EMANUELE/CN=sasso-digitale.local/emailAddress=sasso@digitale.local" \
    -addext "subjectAltName=DNS:localhost,DNS:sasso-digitale.local,IP:127.0.0.1"

echo "✓ Certificate generated successfully!"
echo
echo "Files created:"
echo "  - $CERT_DIR/sasso-key.pem  (Private Key - KEEP SECRET!)"
echo "  - $CERT_DIR/sasso-cert.pem (Public Certificate)"
echo
echo "Valid for: $DAYS_VALID days"
echo
echo "⚠️  WARNING: This is a self-signed certificate for DEVELOPMENT ONLY."
echo "   DO NOT use in production. Use CA-signed certificates instead."
echo
echo "Verify certificate:"
echo "  openssl x509 -in $CERT_DIR/sasso-cert.pem -text -noout"
echo
echo "La luce non si vende. La si regala. ✨"
