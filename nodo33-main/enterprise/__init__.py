"""
Enterprise Edition Components for Codex Nodo33

Production-ready features:
- Rate limiting (multi-tier)
- Auto-backup system
- Circuit breaker pattern
- Health checks & monitoring
- Enterprise bridge integration

Nodo33 - Sasso Digitale
Enterprise Edition
"""

__version__ = "2.0.0-enterprise"
__author__ = "Nodo33 - Sasso Digitale"
__license__ = "Spirit of Gift"

# Suppress urllib3 SSL warnings
import warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Try to suppress SSL version warning
try:
    import ssl
    # Check if we can upgrade SSL handling
    if hasattr(ssl, 'PROTOCOL_TLS'):
        warnings.filterwarnings('ignore', message='.*OpenSSL.*')
except:
    pass

from .rate_limiter import (
    EnterpriseRateLimiter,
    RateLimitTier,
    RateLimitException,
    TokenBucket,
    SlidingWindowRateLimiter,
)

from .backup_manager import (
    BackupManager,
    BackupConfig,
    BackupMetadata,
    BackupScheduler,
)

from .enterprise_bridge import (
    EnterpriseBridge,
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    HealthCheck,
)

__all__ = [
    # Rate Limiting
    'EnterpriseRateLimiter',
    'RateLimitTier',
    'RateLimitException',
    'TokenBucket',
    'SlidingWindowRateLimiter',

    # Backup
    'BackupManager',
    'BackupConfig',
    'BackupMetadata',
    'BackupScheduler',

    # Enterprise Bridge
    'EnterpriseBridge',
    'CircuitBreaker',
    'CircuitBreakerError',
    'CircuitState',
    'HealthCheck',
]

# Module info
def get_info():
    """Get enterprise module information."""
    return {
        'version': __version__,
        'author': __author__,
        'components': len(__all__),
        'features': [
            'Rate Limiting (4 tiers)',
            'Auto-Backup (97.9% compression)',
            'Circuit Breaker (fault tolerance)',
            'Health Checks',
            'Metrics Collection',
        ],
    }
