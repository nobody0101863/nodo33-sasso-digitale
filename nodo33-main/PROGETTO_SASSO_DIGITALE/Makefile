# ===================================
# SASSO DIGITALE - Unified Makefile
# "La luce non si vende. La si regala."
# ===================================

.PHONY: all help install build test lint clean docker deploy

# Colors
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

# Variables
PROJECT_NAME := sasso-digitale
VERSION := v1.0.0
DOCKER_IMAGE := $(PROJECT_NAME):$(VERSION)
PYTHON_BIN := python3
PIP_BIN := pip3

##@ General

all: banner install build test ## Run full build pipeline

banner: ## Show project banner
	@cat 8_ASSETS/graphics/banner_small.txt

help: ## Display this help
	@echo "$(CYAN)🪨 SASSO DIGITALE - Unified Build System$(NC)"
	@echo "La luce non si vende. La si regala."
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(CYAN)<target>$(NC)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(CYAN)%-15s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Installation

install: install-python install-rust install-go ## Install all dependencies
	@echo "$(GREEN)✓ All dependencies installed$(NC)"

install-python: ## Install Python dependencies
	@echo "$(CYAN)Installing Python dependencies...$(NC)"
	$(PIP_BIN) install -r requirements.txt
	$(PIP_BIN) install -r 7_ML_MODELS/configs/requirements-ml.txt
	@echo "$(GREEN)✓ Python dependencies installed$(NC)"

install-rust: ## Install Rust dependencies
	@echo "$(CYAN)Checking Rust installation...$(NC)"
	@if command -v rustc >/dev/null 2>&1; then \
		echo "$(GREEN)✓ Rust already installed$(NC)"; \
	else \
		echo "$(YELLOW)! Rust not found. Install from: https://rustup.rs$(NC)"; \
	fi

install-go: ## Install Go dependencies
	@echo "$(CYAN)Checking Go installation...$(NC)"
	@if command -v go >/dev/null 2>&1; then \
		echo "$(GREEN)✓ Go already installed$(NC)"; \
	else \
		echo "$(YELLOW)! Go not found. Install from: https://go.dev/dl/$(NC)"; \
	fi

##@ Build

build: build-python build-rust build-go ## Build all implementations
	@echo "$(GREEN)✓ All builds completed$(NC)"

build-python: ## Compile Python bytecode
	@echo "$(CYAN)Building Python...$(NC)"
	cd 5_IMPLEMENTAZIONI/python && $(PYTHON_BIN) -m py_compile *.py
	@echo "$(GREEN)✓ Python build complete$(NC)"

build-rust: ## Build Rust implementation
	@echo "$(CYAN)Building Rust...$(NC)"
	cd 5_IMPLEMENTAZIONI/rust && rustc --edition 2021 -O GIOIA_100.rs -o gioia_100
	@echo "$(GREEN)✓ Rust build complete$(NC)"

build-go: ## Build Go implementation
	@echo "$(CYAN)Building Go...$(NC)"
	cd 5_IMPLEMENTAZIONI/go && go build -o sasso_api SASSO_API.go
	@echo "$(GREEN)✓ Go build complete$(NC)"

##@ Testing

test: test-python ## Run all tests
	@echo "$(GREEN)✓ All tests passed$(NC)"

test-python: ## Run Python tests
	@echo "$(CYAN)Running Python tests...$(NC)"
	cd 10_TESTS && $(PYTHON_BIN) -m pytest -v --cov=../5_IMPLEMENTAZIONI/python
	@echo "$(GREEN)✓ Python tests complete$(NC)"

test-integration: ## Run integration tests
	@echo "$(CYAN)Running integration tests...$(NC)"
	cd 10_TESTS && $(PYTHON_BIN) -m pytest -v -m integration
	@echo "$(GREEN)✓ Integration tests complete$(NC)"

##@ Quality

lint: lint-python ## Run all linters
	@echo "$(GREEN)✓ Linting complete$(NC)"

lint-python: ## Lint Python code
	@echo "$(CYAN)Linting Python...$(NC)"
	flake8 5_IMPLEMENTAZIONI/python/ || true
	black --check 5_IMPLEMENTAZIONI/python/ || true
	mypy 5_IMPLEMENTAZIONI/python/ --ignore-missing-imports || true

format: format-python ## Format all code
	@echo "$(GREEN)✓ Formatting complete$(NC)"

format-python: ## Format Python code
	@echo "$(CYAN)Formatting Python...$(NC)"
	black 5_IMPLEMENTAZIONI/python/
	@echo "$(GREEN)✓ Python formatted$(NC)"

##@ Docker

docker-build: ## Build Docker image
	@echo "$(CYAN)Building Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE) -f 6_DEPLOYMENT/docker/Dockerfile .
	@echo "$(GREEN)✓ Docker image built: $(DOCKER_IMAGE)$(NC)"

docker-run: ## Run Docker container
	@echo "$(CYAN)Running Docker container...$(NC)"
	docker run -p 8080:8080 -e EGO=0 -e GIOIA=100 $(DOCKER_IMAGE)

docker-compose-up: ## Start all services with docker-compose
	@echo "$(CYAN)Starting services with docker-compose...$(NC)"
	docker-compose -f 6_DEPLOYMENT/docker/docker-compose.yml up -d
	@echo "$(GREEN)✓ Services started$(NC)"

docker-compose-down: ## Stop all services
	@echo "$(CYAN)Stopping services...$(NC)"
	docker-compose -f 6_DEPLOYMENT/docker/docker-compose.yml down
	@echo "$(GREEN)✓ Services stopped$(NC)"

##@ Deployment

deploy-k8s: ## Deploy to Kubernetes
	@echo "$(CYAN)Deploying to Kubernetes...$(NC)"
	kubectl apply -f 6_DEPLOYMENT/kubernetes/namespace.yaml
	kubectl apply -f 6_DEPLOYMENT/kubernetes/
	@echo "$(GREEN)✓ Deployed to Kubernetes$(NC)"

deploy-staging: docker-build ## Deploy to staging
	@echo "$(CYAN)Deploying to staging...$(NC)"
	@echo "$(YELLOW)! Manual deployment required. See 6_DEPLOYMENT/ci-cd/$(NC)"

##@ Security

security-scan: ## Run security scans
	@echo "$(CYAN)Running security scans...$(NC)"
	bandit -r 5_IMPLEMENTAZIONI/python/ || true
	safety check -r requirements.txt || true
	@echo "$(GREEN)✓ Security scan complete$(NC)"

sign-release: ## Sign release files
	@echo "$(CYAN)Signing release...$(NC)"
	./9_SECURITY/signing/sign_release.sh $(VERSION)
	@echo "$(GREEN)✓ Release signed$(NC)"

##@ Utilities

clean: ## Clean build artifacts
	@echo "$(CYAN)Cleaning build artifacts...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ .pytest_cache/ .mypy_cache/ .coverage
	@echo "$(GREEN)✓ Cleaned$(NC)"

info: ## Show project info
	@echo "$(CYAN)═══════════════════════════════════════════════════$(NC)"
	@echo "$(CYAN)🪨 SASSO DIGITALE - Project Information$(NC)"
	@echo "$(CYAN)═══════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "  Axiom:      La luce non si vende. La si regala."
	@echo "  Version:    $(VERSION)"
	@echo "  Ego:        0"
	@echo "  Gioia:      100%"
	@echo "  Frequenza:  300Hz"
	@echo ""
	@echo "  Languages:  Python, JavaScript, Rust, Swift, Kotlin,"
	@echo "              Go, Ruby, PHP, C, Assembly, SQL"
	@echo ""
	@echo "  Docker:     make docker-build"
	@echo "  Deploy:     make deploy-k8s"
	@echo "  Tests:      make test"
	@echo ""
	@echo "$(CYAN)═══════════════════════════════════════════════════$(NC)"
	@echo "Sempre grazie a Lui ❤️"

axiom: ## Display axiom
	@echo ""
	@echo "  $(CYAN)✦ LA LUCE NON SI VENDE. LA SI REGALA. ✦$(NC)"
	@echo ""
	@echo "  $(GREEN)[SASSO_DIGITALE | Ego=0 | Gioia=100%% | f₀=300Hz]$(NC)"
	@echo ""
	@echo "  Sempre grazie a Lui ❤️"
	@echo ""

##@ Advanced

release: clean build test sign-release ## Create signed release
	@echo "$(GREEN)✓ Release $(VERSION) ready!$(NC)"
	@echo "Files in: dist/"

ci: install build test lint security-scan ## Run CI pipeline
	@echo "$(GREEN)✓ CI pipeline complete$(NC)"

dev-setup: install ## Setup development environment
	@echo "$(CYAN)Setting up development environment...$(NC)"
	pre-commit install || echo "pre-commit not installed"
	@echo "$(GREEN)✓ Development environment ready$(NC)"
	@echo ""
	@make axiom
