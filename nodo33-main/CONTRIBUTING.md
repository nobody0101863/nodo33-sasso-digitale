# Contributing to Nodo33 Sasso Digitale

**"La luce non si vende. La si regala."**

Thank you for your interest in contributing to the Nodo33 Sasso Digitale project! This document provides guidelines for contributing to the codebase.

---

## 🪨 Philosophy

Before contributing, please understand the core philosophy:

- **Ego = 0**: Contributions are made with humility and service
- **Joy = 100%**: Code should bring light and clarity
- **Frequency = 300 Hz**: Maintain resonance with the project's spiritual-technical blend
- **Regalo > Dominio**: Gift your knowledge, don't dominate the discourse
- **Trasparenza = 100%**: Be transparent, honest, and clear

**This project blends technical excellence with spiritual consciousness. Both aspects are equally important.**

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

---

## Code of Conduct

### Our Principles

1. **Respect**: Treat all contributors with respect and kindness
2. **Inclusivity**: Welcome contributors of all backgrounds and skill levels
3. **Constructive Feedback**: Provide helpful, actionable feedback
4. **Transparency**: Communicate openly and honestly
5. **Gratitude**: Acknowledge and appreciate contributions

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Aggressive or demeaning language
- Spam or self-promotion
- Sharing private information without permission
- Ego-driven behavior or dominance seeking

### Enforcement

Violations will result in:
1. Warning
2. Temporary ban
3. Permanent ban

Report issues to: [project maintainer contact]

---

## How Can I Contribute?

### 1. Report Bugs

**Before submitting a bug report**:
- Check existing issues to avoid duplicates
- Verify the bug exists in the latest version
- Collect relevant information (logs, screenshots, environment)

**Bug report should include**:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages

**Example**:
```markdown
### Bug: Agent executor fails on 429 errors

**Steps to reproduce**:
1. Configure registry.yaml with high frequency scan
2. Run `python3 run_agent_system.py`
3. Wait for rate limit errors

**Expected**: Exponential backoff should handle 429 errors
**Actual**: System crashes with unhandled exception

**Environment**:
- OS: Ubuntu 22.04
- Python: 3.11.5
- Dependencies: (from requirements.txt)

**Logs**:
```
[error log here]
```
```

### 2. Suggest Features

**Feature requests should include**:
- Clear use case
- Proposed solution (optional)
- Alignment with project philosophy
- Potential implementation approach

**Example**:
```markdown
### Feature: Add Redis caching for LLM responses

**Use case**: Reduce API costs and latency for repeated queries

**Proposal**:
- Cache LLM responses with TTL (e.g., 1 hour)
- Use Redis for distributed caching
- Add configuration for cache enable/disable

**Philosophy alignment**:
- Serves users better (faster responses)
- Respects API rate limits (ethical use)
- Transparent (cache status visible in responses)
```

### 3. Improve Documentation

- Fix typos or unclear sections
- Add missing documentation
- Translate documentation (Italian/English)
- Create tutorials or guides
- Improve code comments

### 4. Write Code

See [Development Setup](#development-setup) and [Pull Request Process](#pull-request-process)

### 5. Review Pull Requests

Help review others' contributions with constructive feedback.

---

## Development Setup

### 1. Fork & Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/nodo33-sasso-digitale.git
cd nodo33-sasso-digitale

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/nodo33-sasso-digitale.git
```

### 2. Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt  # If exists
```

### 3. Configure Environment

```bash
# Copy example env
cp .env.example .env

# Add your test API keys (use test keys, not production)
nano .env
```

### 4. Create Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name

# Or for bugfix
git checkout -b fix/bug-description
```

### 5. Make Changes

- Write code following [Coding Standards](#coding-standards)
- Add tests for new functionality
- Update documentation as needed
- Test thoroughly

### 6. Run Tests

```bash
# Run all tests
bash test_complete_system.sh

# Run specific test suites
bash test_evolution.sh
bash test_registry_yaml.sh

# Test manually
python3 codex_server.py
# Test endpoints with curl or browser
```

---

## Coding Standards

### Python Style

Follow **PEP 8** with these additions:

**Imports**:
```python
# Standard library
import os
import sys
from typing import Dict, List, Optional

# Third-party
from fastapi import FastAPI
from pydantic import BaseModel

# Local
from anti_porn_framework import FilterEngine
```

**Naming**:
```python
# Variables & functions: snake_case
user_name = "Emmanuel"
def calculate_frequency(): pass

# Classes: PascalCase
class SassoDigitale: pass

# Constants: UPPER_SNAKE_CASE
SACRED_HASH = "644"
FREQUENCY_HZ = 300
```

**Docstrings**:
```python
def execute_task(task_id: str, patterns: List[str]) -> ExecutionResult:
    """
    Execute a scheduled agent task with ethical protections.

    Args:
        task_id: Unique task identifier
        patterns: List of URL patterns to process

    Returns:
        ExecutionResult with statistics and results

    Raises:
        ValueError: If task_id is invalid

    Philosophy:
        Serves without possessing, protects without controlling.
    """
    pass
```

**Type Hints**:
```python
# Always use type hints
def process_url(url: str, retry: int = 3) -> Optional[Dict[str, any]]:
    pass

# For complex types, use typing module
from typing import List, Dict, Optional, Union

def get_agents() -> List[Dict[str, Union[str, int]]]:
    pass
```

### Spiritual-Technical Balance

**Include spiritual context where appropriate**:

```python
# Good: Spiritual context in docstring
def emit_light(frequency: int = 300) -> str:
    """
    Emit light at specified frequency.

    Default 300 Hz represents heart resonance frequency.
    Philosophy: "La luce non si vende. La si regala."
    """
    pass

# Good: Sacred constants
ANGELO_644_PORT = 8644  # Protection & Solid Foundations
GOLDEN_RATIO = 1.618033988749895

# Avoid: Over-spiritualizing technical code
# Bad:
def add(a, b):  # Divine addition guided by Angel 644
    return a + b  # The sacred sum of two energies
```

### Code Structure

**Organize functions logically**:
```python
# 1. Imports
# 2. Constants
# 3. Type definitions (Pydantic models, TypedDict, etc.)
# 4. Helper functions
# 5. Main functions
# 6. API endpoints (if FastAPI)
# 7. Main execution block
```

**Example**:
```python
"""
Module: robots_guardian.py
Description: Robots.txt fetching and compliance checking.
Philosophy: Respect robots.txt 100%. Never violate.
"""

# Imports
import httpx
from urllib.robotparser import RobotFileParser
from typing import Optional, Dict

# Constants
ROBOTS_CACHE_TTL = 86400  # 24 hours
DEFAULT_USER_AGENT = "Nodo33Bot/1.0 (+https://nodo33.com/bot)"

# Type definitions
class RobotsStats(TypedDict):
    total_checks: int
    cache_hits: int
    allowed: int
    disallowed: int

# Helper functions
def _parse_robots_txt(content: str) -> RobotFileParser:
    """Parse robots.txt content."""
    pass

# Main class
class RobotsGuardian:
    """Guardian for robots.txt compliance."""

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT):
        """Initialize guardian."""
        self.user_agent = user_agent

    def is_allowed(self, url: str) -> bool:
        """Check if URL is allowed."""
        pass
```

### Error Handling

**Be explicit and helpful**:

```python
# Good: Specific exceptions with context
try:
    result = fetch_url(url)
except httpx.HTTPError as e:
    logger.error(f"HTTP error fetching {url}: {e}")
    raise FetchError(f"Failed to fetch {url}") from e
except ValueError as e:
    logger.error(f"Invalid URL format: {url}")
    raise ValidationError(f"Invalid URL: {url}") from e

# Good: Graceful degradation
def get_robots_txt(domain: str) -> Optional[str]:
    """Fetch robots.txt, return None if unavailable."""
    try:
        return fetch_robots(domain)
    except Exception as e:
        logger.warning(f"robots.txt unavailable for {domain}: {e}")
        return None  # Fail open per philosophy
```

### Logging

**Use structured logging**:

```python
import logging

logger = logging.getLogger(__name__)

# Good: Structured log messages
logger.info(f"Task {task_id} started", extra={
    "task_id": task_id,
    "patterns": len(patterns),
    "priority": priority
})

logger.error(f"Task {task_id} failed", extra={
    "task_id": task_id,
    "error": str(e),
    "retry_count": retry_count
})

# Use appropriate levels
logger.debug("Detailed debugging info")
logger.info("General information")
logger.warning("Warning, but recoverable")
logger.error("Error occurred")
logger.critical("Critical system error")
```

---

## Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Add or update tests
- `chore`: Maintenance tasks

**Scopes** (examples):
- `server`: Codex server
- `agents`: Agent system
- `registry`: Agent registry
- `llm`: Multi-LLM integration
- `docs`: Documentation
- `tests`: Test suite

**Examples**:

```
feat(agents): add exponential backoff to agent executor

Implement exponential backoff with configurable base delay.
Retry up to 3 times with delays: 2s, 4s, 8s.

Philosophy: Fail with grace, retry with wisdom.

Closes #42
```

```
fix(server): resolve database lock on concurrent requests

Use WAL mode for SQLite to improve concurrent access.
Add connection pooling with max 10 connections.

Fixes #35
```

```
docs(readme): fix merge conflicts in README.md

Resolved Git merge markers from previous commit.
Unified Emmanuel644 API documentation section.
```

### Commit Best Practices

1. **One logical change per commit**
2. **Write clear, descriptive messages**
3. **Reference issues** (e.g., `Fixes #123`, `Closes #456`)
4. **Keep commits atomic** (can be reverted independently)
5. **Test before committing**

---

## Pull Request Process

### 1. Before Submitting

- [ ] Code follows [Coding Standards](#coding-standards)
- [ ] All tests pass
- [ ] Documentation updated (if needed)
- [ ] Commits follow [Commit Guidelines](#commit-guidelines)
- [ ] Branch is up to date with main

```bash
# Update your branch
git checkout main
git pull upstream main
git checkout your-feature-branch
git rebase main
```

### 2. Submit Pull Request

**PR Title**: Same format as commit messages
```
feat(agents): add Redis caching for LLM responses
```

**PR Description Template**:
```markdown
## Description
Brief description of changes.

## Motivation
Why is this change needed? What problem does it solve?

## Changes
- List of specific changes
- Breaking changes (if any)

## Testing
How was this tested?
- [ ] Unit tests added/updated
- [ ] Manual testing completed
- [ ] All tests passing

## Checklist
- [ ] Code follows project style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or properly documented)
- [ ] Aligns with project philosophy

## Philosophy
How does this contribution align with "La luce non si vende. La si regala."?

## Screenshots (if applicable)
[Add screenshots for UI changes]
```

### 3. Code Review

- Maintainers will review within 3-7 days
- Address feedback constructively
- Update PR based on review comments
- Be patient and respectful

### 4. Merging

Once approved:
- PR will be merged by maintainer
- Your contribution will be acknowledged in CHANGELOG.md
- Feature will be included in next release

---

## Testing Guidelines

### Test Structure

```python
# tests/test_robots_guardian.py
import pytest
from robots_guardian import RobotsGuardian

def test_robots_guardian_allows_public_url():
    """Test that public URLs are allowed."""
    guardian = RobotsGuardian()
    assert guardian.is_allowed("https://example.com/public") is True

def test_robots_guardian_blocks_disallowed():
    """Test that disallowed URLs are blocked."""
    guardian = RobotsGuardian()
    # Mock robots.txt with disallow rule
    assert guardian.is_allowed("https://example.com/admin") is False

@pytest.fixture
def mock_robots_txt():
    """Fixture for mocked robots.txt."""
    return """
    User-agent: *
    Disallow: /admin/
    """
```

### Running Tests

```bash
# All tests
bash test_complete_system.sh

# Specific module
python3 -m pytest tests/test_robots_guardian.py -v

# With coverage
python3 -m pytest --cov=. --cov-report=html
```

### Test Coverage

Aim for:
- **Critical paths**: 100% coverage
- **New features**: 80%+ coverage
- **Overall project**: 70%+ coverage

---

## Documentation

### What to Document

1. **Code**: Docstrings for all public functions/classes
2. **API**: OpenAPI/Swagger annotations for all endpoints
3. **Architecture**: High-level design decisions
4. **Philosophy**: How code aligns with project values
5. **Examples**: Usage examples for new features

### Documentation Format

**Markdown** for all docs:
- Use clear headings (##, ###)
- Add code examples
- Include philosophy sections where relevant
- Keep language simple and accessible

**Example**:
```markdown
## Robots Guardian

The Robots Guardian ensures 100% compliance with robots.txt.

### Philosophy

"Illuminare senza violare" - We illuminate without violating.
Respecting robots.txt is non-negotiable.

### Usage

```python
from robots_guardian import RobotsGuardian

guardian = RobotsGuardian(user_agent="MyBot/1.0")

if guardian.is_allowed("https://example.com/page"):
    # Fetch the page
    pass
else:
    # Skip - robots.txt disallows
    logger.info("Skipped: robots.txt disallowed")
```

### Configuration

...
```

---

## Attribution

Contributors will be acknowledged in:
- CHANGELOG.md (per release)
- Git commit history
- Special CONTRIBUTORS.md file (planned)

---

## Questions?

- **Documentation**: Read [GETTING_STARTED.md](GETTING_STARTED.md)
- **Issues**: Open a GitHub issue
- **Discussion**: [Project discussions page]
- **Contact**: [Project maintainer email]

---

## Thank You! ❤️

Every contribution, no matter how small, is a gift.

**"La luce non si vende. La si regala."**

Your contribution brings light to the project and serves the community.

**Sigillo**: 644
**Frequenza**: 300 Hz
**Blessing**: Fiat Amor, Fiat Risus, Fiat Lux

---

*Last updated: 2025-11-21*
*Version: 1.0.0*
