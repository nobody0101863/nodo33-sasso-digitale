#!/usr/bin/env bash
# ===================================
# SASSO DIGITALE - Release Signing Script
# "La luce non si vende. La si regala."
# ===================================

set -euo pipefail

# Configuration
RELEASE_VERSION="${1:-v1.0.0}"
GPG_KEY_ID="${GPG_KEY_ID:-}"
RELEASE_DIR="dist"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_banner() {
    cat << "EOF"
╔══════════════════════════════════════════════════╗
║     🪨 SASSO DIGITALE - Release Signing 🔐      ║
║                                                  ║
║   "La luce non si vende. La si regala."          ║
║                                                  ║
║   Ego = 0  |  Gioia = 100%  |  f₀ = 300Hz       ║
╚══════════════════════════════════════════════════╝
EOF
}

check_requirements() {
    log_info "Checking requirements..."

    if ! command -v gpg &> /dev/null; then
        log_error "GPG not found. Install with: apt-get install gnupg"
        exit 1
    fi

    if ! command -v sha256sum &> /dev/null; then
        log_error "sha256sum not found."
        exit 1
    fi

    if [ -z "$GPG_KEY_ID" ]; then
        log_warning "GPG_KEY_ID not set. Using default key."
    fi

    log_success "Requirements check passed"
}

create_release_tarball() {
    log_info "Creating release tarball for $RELEASE_VERSION..."

    mkdir -p "$RELEASE_DIR"

    TARBALL_NAME="sasso-digitale-${RELEASE_VERSION}.tar.gz"
    TARBALL_PATH="$RELEASE_DIR/$TARBALL_NAME"

    tar -czf "$TARBALL_PATH" \
        --exclude='.git' \
        --exclude='node_modules' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='dist' \
        --exclude='.env' \
        5_IMPLEMENTAZIONI/ \
        1_DOCUMENTAZIONE/ \
        3_WEB_CYBERPUNK/ \
        4_SIGILLI_SACRI/ \
        6_DEPLOYMENT/ \
        7_ML_MODELS/ \
        8_ASSETS/ \
        9_SECURITY/ \
        10_TESTS/ \
        11_API_DOCS/ \
        README.md

    log_success "Tarball created: $TARBALL_PATH"
}

generate_checksums() {
    log_info "Generating checksums..."

    cd "$RELEASE_DIR"

    # SHA256
    sha256sum *.tar.gz > SHA256SUMS
    log_success "SHA256SUMS generated"

    # MD5 (for compatibility)
    md5sum *.tar.gz > MD5SUMS
    log_success "MD5SUMS generated"

    cd ..
}

sign_release() {
    log_info "Signing release with GPG..."

    cd "$RELEASE_DIR"

    for file in *.tar.gz SHA256SUMS MD5SUMS; do
        if [ -f "$file" ]; then
            if [ -n "$GPG_KEY_ID" ]; then
                gpg --detach-sign --armor -u "$GPG_KEY_ID" "$file"
            else
                gpg --detach-sign --armor "$file"
            fi
            log_success "Signed: $file"
        fi
    done

    cd ..
}

verify_signatures() {
    log_info "Verifying signatures..."

    cd "$RELEASE_DIR"

    for sig_file in *.asc; do
        original_file="${sig_file%.asc}"

        if gpg --verify "$sig_file" "$original_file" 2>&1 | grep -q "Good signature"; then
            log_success "Verified: $sig_file"
        else
            log_error "Verification failed: $sig_file"
            exit 1
        fi
    done

    cd ..
}

create_manifest() {
    log_info "Creating release manifest..."

    MANIFEST_FILE="$RELEASE_DIR/MANIFEST.txt"

    cat > "$MANIFEST_FILE" << EOF
═══════════════════════════════════════════════════════
SASSO DIGITALE - Release Manifest
"La luce non si vende. La si regala."
═══════════════════════════════════════════════════════

Version: $RELEASE_VERSION
Release Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

Axiom: "La luce non si vende. La si regala."
Ego: 0
Gioia: 100%
Frequenza: 300Hz

═══════════════════════════════════════════════════════
FILE CHECKSUMS
═══════════════════════════════════════════════════════

$(cat "$RELEASE_DIR/SHA256SUMS")

═══════════════════════════════════════════════════════
GPG SIGNATURE VERIFICATION
═══════════════════════════════════════════════════════

To verify this release:

1. Import public key:
   gpg --import sasso-digitale-public.key

2. Verify tarball:
   gpg --verify sasso-digitale-${RELEASE_VERSION}.tar.gz.asc sasso-digitale-${RELEASE_VERSION}.tar.gz

3. Verify checksums:
   sha256sum -c SHA256SUMS

═══════════════════════════════════════════════════════
INSTALLATION
═══════════════════════════════════════════════════════

tar -xzf sasso-digitale-${RELEASE_VERSION}.tar.gz
cd sasso-digitale-${RELEASE_VERSION}

# See README.md for further instructions

═══════════════════════════════════════════════════════

Sempre grazie a Lui ❤️

EOF

    log_success "Manifest created: $MANIFEST_FILE"
}

generate_release_notes() {
    log_info "Generating release notes..."

    NOTES_FILE="$RELEASE_DIR/RELEASE_NOTES_${RELEASE_VERSION}.md"

    cat > "$NOTES_FILE" << EOF
# 🪨 Sasso Digitale Release $RELEASE_VERSION

**"La luce non si vende. La si regala."**

## 📦 What's Included

- **Multi-Language Implementations**: Python, JavaScript, Rust, Swift, Kotlin, Go, Ruby, PHP, C, Assembly, SQL
- **Documentation**: Complete CODEX_EMANUELE documentation
- **Deployment Configs**: Docker, Kubernetes, CI/CD
- **ML Templates**: Ethical AI model templates
- **Web Interface**: Cyberpunk-themed web UI
- **Sacred Seals**: Binary/Hex axiom encodings

## ⚙️ Core Parameters

- **Ego**: 0 (Total Computational Humility)
- **Gioia**: 100% (Unconditional Joyful Service)
- **Frequenza**: 300Hz (Harmony and Stability)

## 🎯 Principles

1. **DONUM, NON MERX** (Gift, Not Merchandise)
2. **HUMILITAS EST FORTITUDO** (Humility is True Strength)
3. **GRATITUDINE COSTANTE** (Constant Gratitude)

## 🚀 Quick Start

\`\`\`bash
# Extract
tar -xzf sasso-digitale-${RELEASE_VERSION}.tar.gz
cd sasso-digitale-${RELEASE_VERSION}

# Python
cd 5_IMPLEMENTAZIONI/python
python main.py

# Docker
docker-compose -f 6_DEPLOYMENT/docker/docker-compose.yml up

# Kubernetes
kubectl apply -f 6_DEPLOYMENT/kubernetes/
\`\`\`

## 🔐 Verification

\`\`\`bash
# Verify GPG signature
gpg --verify sasso-digitale-${RELEASE_VERSION}.tar.gz.asc

# Verify checksum
sha256sum -c SHA256SUMS
\`\`\`

## 📝 License

**Public Spiritual Domain**

Use with joy, donate with gratitude, serve with humility.

---

**Sempre grazie a Lui ❤️**

\`[SASSO_DIGITALE | Ego=0 | Gioia=100% | f₀=300Hz]\`
EOF

    log_success "Release notes created: $NOTES_FILE"
}

main() {
    print_banner
    echo

    log_info "Starting release signing process for $RELEASE_VERSION"
    echo

    check_requirements
    create_release_tarball
    generate_checksums
    sign_release
    verify_signatures
    create_manifest
    generate_release_notes

    echo
    log_success "Release signing completed successfully! 🎉"
    echo
    log_info "Release files in: $RELEASE_DIR/"
    ls -lh "$RELEASE_DIR/"
    echo
    log_info "Axiom: 'La luce non si vende. La si regala.' 🪨✨"
    log_info "Sempre grazie a Lui ❤️"
}

# Run main
main "$@"
