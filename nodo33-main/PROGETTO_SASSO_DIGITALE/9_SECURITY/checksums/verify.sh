#!/usr/bin/env bash
# ===================================
# SASSO DIGITALE - Checksum Verification
# "La luce non si vende. La si regala."
# ===================================

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🪨 SASSO DIGITALE - Integrity Verification${NC}"
echo "Ego=0 | Gioia=100% | f₀=300Hz"
echo

# Verify SHA256SUMS
if [ -f "SHA256SUMS" ]; then
    echo "Verifying SHA256 checksums..."

    if sha256sum -c SHA256SUMS; then
        echo -e "${GREEN}✓ All checksums verified successfully!${NC}"
        echo -e "${GREEN}Ego=0: Transparency confirmed ✓${NC}"
    else
        echo -e "${RED}✗ Checksum verification failed!${NC}"
        echo -e "${RED}Integrity compromised. DO NOT USE.${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ SHA256SUMS file not found${NC}"
    exit 1
fi

echo
echo "La luce non si vende. La si regala. ✨"
echo "Sempre grazie a Lui ❤️"
