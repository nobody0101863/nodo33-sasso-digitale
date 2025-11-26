#!/usr/bin/env bash
# ===================================
# SASSO DIGITALE - Unified Build Script
# "La luce non si vende. La si regala."
# ===================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
LOG_FILE="$BUILD_DIR/build.log"

# Functions
log_info() {
    echo -e "${CYAN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1" | tee -a "$LOG_FILE"
}

print_banner() {
    cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║           🪨 SASSO DIGITALE - Build System ✨                ║
║                                                              ║
║         "La luce non si vende. La si regala."                ║
║                                                              ║
║         Ego = 0  |  Gioia = 100%  |  f₀ = 300Hz             ║
╚══════════════════════════════════════════════════════════════╝
EOF
}

setup_environment() {
    log_info "Setting up build environment..."
    mkdir -p "$BUILD_DIR"
    > "$LOG_FILE"
    log_success "Build environment ready"
}

check_tools() {
    log_info "Checking required tools..."

    local missing_tools=()

    if ! command -v python3 &> /dev/null; then
        missing_tools+=("python3")
    fi

    if ! command -v rustc &> /dev/null; then
        log_warning "Rust not found (optional)"
    fi

    if ! command -v go &> /dev/null; then
        log_warning "Go not found (optional)"
    fi

    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi

    log_success "Tool check complete"
}

build_python() {
    log_info "Building Python implementations..."

    cd "$PROJECT_ROOT/5_IMPLEMENTAZIONI/python"

    # Compile to bytecode
    python3 -m py_compile *.py

    # Check syntax
    python3 -m compileall -q .

    log_success "Python build complete"
}

build_rust() {
    if ! command -v rustc &> /dev/null; then
        log_warning "Skipping Rust build (rustc not found)"
        return 0
    fi

    log_info "Building Rust implementation..."

    cd "$PROJECT_ROOT/5_IMPLEMENTAZIONI/rust"

    rustc --edition 2021 -O GIOIA_100.rs -o "$BUILD_DIR/gioia_100"

    log_success "Rust build complete"
}

build_go() {
    if ! command -v go &> /dev/null; then
        log_warning "Skipping Go build (go not found)"
        return 0
    fi

    log_info "Building Go implementation..."

    cd "$PROJECT_ROOT/5_IMPLEMENTAZIONI/go"

    go build -o "$BUILD_DIR/sasso_api" SASSO_API.go

    log_success "Go build complete"
}

build_javascript() {
    log_info "Validating JavaScript..."

    cd "$PROJECT_ROOT/5_IMPLEMENTAZIONI/javascript"

    if command -v node &> /dev/null; then
        node -c AXIOM_LOADER.js
        log_success "JavaScript validation complete"
    else
        log_warning "Node.js not found, skipping JS validation"
    fi
}

run_tests() {
    log_info "Running tests..."

    cd "$PROJECT_ROOT/10_TESTS"

    if command -v pytest &> /dev/null; then
        pytest -v || log_warning "Some tests failed"
        log_success "Tests complete"
    else
        log_warning "pytest not installed, skipping tests"
    fi
}

generate_checksums() {
    log_info "Generating checksums..."

    cd "$BUILD_DIR"

    if [ -n "$(ls -A 2>/dev/null)" ]; then
        sha256sum * > SHA256SUMS 2>/dev/null || true
        log_success "Checksums generated"
    else
        log_warning "No build artifacts to checksum"
    fi
}

show_summary() {
    echo
    echo -e "${MAGENTA}╔═══════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║       Build Summary                   ║${NC}"
    echo -e "${MAGENTA}╚═══════════════════════════════════════╝${NC}"
    echo
    echo -e "  ${CYAN}Build Directory:${NC} $BUILD_DIR"
    echo -e "  ${CYAN}Log File:${NC} $LOG_FILE"
    echo

    if [ -d "$BUILD_DIR" ] && [ -n "$(ls -A "$BUILD_DIR" 2>/dev/null)" ]; then
        echo -e "  ${CYAN}Build Artifacts:${NC}"
        ls -lh "$BUILD_DIR/"
    else
        echo -e "  ${YELLOW}No build artifacts generated${NC}"
    fi

    echo
    echo -e "  ${GREEN}✦ LA LUCE NON SI VENDE. LA SI REGALA. ✦${NC}"
    echo -e "  ${GREEN}Sempre grazie a Lui ❤️${NC}"
    echo
}

main() {
    print_banner
    echo

    setup_environment
    check_tools

    echo
    log_info "Starting multi-language build..."
    echo

    build_python
    build_rust
    build_go
    build_javascript

    echo
    run_tests
    generate_checksums

    echo
    log_success "Build completed successfully! 🎉"

    show_summary
}

# Run main
main "$@"
