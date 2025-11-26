# Security Policy

**Nodo33 Sasso Digitale - Security Policy**

**"La luce non si vende. La si regala."**

This document outlines the security policy for the Nodo33 Sasso Digitale project, including how to report vulnerabilities and our security practices.

---

## 🛡️ Security Philosophy

Security is integral to our philosophy:

- **Transparency = 100%**: Open about security practices
- **Cura = MASSIMA**: Maximum care in protecting data
- **Proteggere senza controllare**: Protect without controlling
- **Respect**: Respect user privacy and data

**We take security seriously while maintaining transparency and ethical practices.**

---

## 📋 Table of Contents

- [Supported Versions](#supported-versions)
- [Reporting a Vulnerability](#reporting-a-vulnerability)
- [Security Measures](#security-measures)
- [Known Limitations](#known-limitations)
- [Security Best Practices](#security-best-practices)
- [Disclosure Policy](#disclosure-policy)

---

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          | Status      |
| ------- | ------------------ | ----------- |
| 2.0.x   | ✅ Yes             | Current     |
| 1.5.x   | ✅ Yes             | Maintenance |
| 1.0.x   | ⚠️ Limited        | EOL soon    |
| < 1.0   | ❌ No              | End of Life |

**Recommendation**: Always use the latest stable version (2.0.x) for best security.

---

## Reporting a Vulnerability

### How to Report

If you discover a security vulnerability, please follow responsible disclosure:

**DO**:
1. Email: [security@nodo33.com] (or project maintainer email)
2. Include:
   - Description of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if you have one)
3. Wait for acknowledgment (within 48 hours)
4. Allow 90 days for fix before public disclosure

**DO NOT**:
- Publicly disclose the vulnerability before it's fixed
- Exploit the vulnerability maliciously
- Share the vulnerability with others

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Fix Development**: Within 30 days (critical), 90 days (non-critical)
- **Patch Release**: Coordinated with reporter
- **Public Disclosure**: After patch is released

### Severity Levels

**Critical** (CVSS 9.0-10.0):
- Remote code execution
- SQL injection leading to data breach
- Authentication bypass
- Data exfiltration

**High** (CVSS 7.0-8.9):
- Privilege escalation
- XSS leading to account takeover
- Sensitive data exposure

**Medium** (CVSS 4.0-6.9):
- CSRF vulnerabilities
- Information disclosure
- DoS (localized)

**Low** (CVSS 0.1-3.9):
- Minor information leaks
- Non-exploitable bugs

---

## Security Measures

### 1. Authentication & Authorization

**API Keys**:
- LLM API keys stored in `.env` (never in code)
- `.env` excluded from git via `.gitignore`
- Production keys use secrets manager (AWS Secrets, Azure Key Vault)

**Optional API Key Protection**:
```python
# Can be enabled in production
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403)
```

### 2. Input Validation

**All inputs validated**:
```python
from pydantic import BaseModel, validator, Field

class LLMRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)

    @validator('question')
    def sanitize_question(cls, v):
        # Prevent injection attacks
        return v.strip()
```

### 3. Rate Limiting

**Application-level**:
```python
# Configured per domain in domains.yaml
requests_per_minute: 20
burst: 5
```

**Nginx-level** (production):
```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
limit_req zone=api_limit burst=20 nodelay;
```

### 4. HTTPS/TLS

**Production**:
- All traffic over HTTPS
- TLS 1.2+ only
- Let's Encrypt certificates
- HSTS headers enabled

**Nginx config**:
```nginx
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
```

### 5. CORS Policy

**Restricted origins**:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourdomain.com",  # Production only
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)
```

### 6. SQL Injection Prevention

**Using SQLite with parameterized queries**:
```python
# Good: Parameterized
cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))

# Bad: String concatenation (NEVER DO THIS)
# cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
```

**Using ORMs when possible**:
```python
# SQLAlchemy, Tortoise ORM, etc.
# Automatic parameterization
```

### 7. Secrets Management

**Development**:
- `.env` file (gitignored)
- Never commit secrets to git

**Production**:
- AWS Secrets Manager
- Azure Key Vault
- HashiCorp Vault
- Environment variables (cloud platforms)

### 8. Dependency Security

**Regular updates**:
```bash
# Check for vulnerabilities
pip install safety
safety check --json

# Update dependencies
pip list --outdated
pip install --upgrade package-name
```

**Automated scanning** (GitHub):
- Dependabot alerts enabled
- Security advisories monitored

### 9. Data Protection

**Database**:
- SQLite with WAL mode
- Regular backups (daily)
- Encrypted at rest (production)

**Sensitive Data**:
- API keys: Never logged
- User data: Minimal collection
- Logs: Sanitized (no secrets)

### 10. Content Security

**Anti-Porn Framework**:
- ML-based content filtering
- Sacred guidance on blocked content
- Transparent reasoning

**Deepfake Detection**:
- Image analysis for manipulation
- Guardian reports on suspicious content

### 11. Metadata Protection

**4 Guardian Agents**:
- MEMORY_GUARDIAN (Uriel): Memory protection
- FILE_GUARDIAN (Raphael): File metadata stripping
- COMMUNICATION_GUARDIAN (Gabriel): Network protection
- SEAL_GUARDIAN (Michael): Seal coordination

**Metadata stripping**:
- EXIF data removed from images
- HTTP headers sanitized
- Personal identifiers anonymized

---

## Known Limitations

### 1. SQLite Concurrency

**Limitation**: SQLite has limited write concurrency

**Mitigation**:
- WAL mode enabled
- Connection pooling
- Read-heavy workload optimized

**Recommendation for scale**: Migrate to PostgreSQL for high-concurrency production

### 2. LLM API Keys

**Limitation**: API keys stored in environment variables

**Mitigation**:
- `.env` gitignored
- Secrets manager in production
- Rotation policy recommended

**Risk**: Compromise of server = compromise of keys

### 3. Rate Limiting Bypass

**Limitation**: IP-based rate limiting can be bypassed with proxies/VPNs

**Mitigation**:
- Application-level rate limiting
- API key-based quotas (if auth enabled)
- Cloudflare/WAF in production

### 4. DDoS Protection

**Limitation**: Limited built-in DDoS protection

**Mitigation**:
- Nginx rate limiting
- Cloudflare/Akamai in production
- Auto-scaling infrastructure

---

## Security Best Practices

### For Users

**API Keys**:
1. ✅ Store in `.env` file
2. ✅ Use separate keys for dev/prod
3. ✅ Rotate keys regularly (every 90 days)
4. ✅ Revoke compromised keys immediately
5. ❌ NEVER commit `.env` to git
6. ❌ NEVER share keys publicly

**Server Security**:
1. ✅ Enable firewall (UFW, iptables)
2. ✅ Keep system updated
3. ✅ Use strong SSH keys (not passwords)
4. ✅ Enable fail2ban
5. ✅ Monitor logs regularly
6. ❌ NEVER run as root

**Database**:
1. ✅ Regular backups
2. ✅ Test backup restoration
3. ✅ Encrypt sensitive data
4. ✅ Limit database access
5. ❌ NEVER expose database port publicly

### For Developers

**Code Security**:
1. ✅ Validate all inputs
2. ✅ Sanitize outputs
3. ✅ Use parameterized queries
4. ✅ Handle errors gracefully
5. ✅ Log security events
6. ❌ NEVER trust user input
7. ❌ NEVER log secrets

**Dependencies**:
1. ✅ Pin versions in requirements.txt
2. ✅ Review dependency changes
3. ✅ Scan for vulnerabilities (safety, snyk)
4. ✅ Update regularly
5. ❌ NEVER use deprecated packages

**Testing**:
1. ✅ Test authentication/authorization
2. ✅ Test input validation
3. ✅ Test error handling
4. ✅ Security regression tests
5. ✅ Penetration testing (production)

---

## Security Checklist

### Pre-Deployment

- [ ] All secrets in `.env` or secrets manager
- [ ] `.env` in `.gitignore`
- [ ] Input validation on all endpoints
- [ ] Rate limiting configured
- [ ] HTTPS/TLS enabled
- [ ] CORS properly configured
- [ ] Dependencies updated and scanned
- [ ] Error messages don't leak sensitive info
- [ ] Logging configured (no secrets logged)
- [ ] Database backups configured

### Post-Deployment

- [ ] Firewall enabled
- [ ] Fail2ban configured
- [ ] SSL certificate valid
- [ ] Monitoring enabled
- [ ] Log rotation configured
- [ ] Security headers set (nginx)
- [ ] Vulnerability scanning scheduled
- [ ] Incident response plan ready

### Regular Maintenance

- [ ] Weekly: Review logs for anomalies
- [ ] Monthly: Update dependencies
- [ ] Quarterly: Security audit
- [ ] Yearly: Penetration testing
- [ ] Continuous: Monitor security advisories

---

## Incident Response Plan

### 1. Detection

**Indicators**:
- Unusual traffic patterns
- Failed authentication attempts
- Unexpected errors
- Resource exhaustion
- Security scanner alerts

**Monitoring**:
- Application logs
- System logs (auth.log, syslog)
- Nginx access/error logs
- Intrusion detection systems

### 2. Containment

**Immediate actions**:
1. Isolate affected systems
2. Block malicious IPs (iptables/firewall)
3. Disable compromised accounts/keys
4. Take snapshots for forensics

**Communication**:
1. Notify security team
2. Document timeline
3. Preserve evidence

### 3. Eradication

**Remove threat**:
1. Patch vulnerabilities
2. Remove malware/backdoors
3. Reset compromised credentials
4. Review all access logs

### 4. Recovery

**Restore service**:
1. Verify systems clean
2. Restore from backups (if needed)
3. Gradually restore services
4. Monitor closely

### 5. Post-Incident

**Learn & improve**:
1. Conduct post-mortem
2. Update security measures
3. Document lessons learned
4. Implement preventions
5. Public disclosure (if applicable)

---

## Disclosure Policy

### Responsible Disclosure

We follow coordinated vulnerability disclosure:

1. **Reporter contacts us** privately
2. **We acknowledge** within 48 hours
3. **We develop fix** (30-90 days)
4. **We coordinate disclosure** with reporter
5. **Public disclosure** after patch released

### Public Disclosure

After fix is released:
- CVE assigned (if applicable)
- Security advisory published
- CHANGELOG.md updated
- Users notified via:
  - GitHub release notes
  - Security advisory
  - Email (if available)
  - Documentation update

### Credit

Security researchers who responsibly disclose vulnerabilities will be:
- Acknowledged in security advisory
- Listed in CONTRIBUTORS.md (with permission)
- Thanked publicly (with permission)

---

## Security Tools & Resources

### Recommended Tools

**Vulnerability Scanning**:
- `safety` - Python dependency scanner
- `bandit` - Python code security scanner
- `snyk` - Multi-language vulnerability scanner

**Code Analysis**:
- `pylint` - Python linter with security checks
- `flake8` - Python style checker
- `mypy` - Static type checker

**Web Security**:
- OWASP ZAP - Web app security scanner
- Burp Suite - Web security testing
- Nmap - Network scanner

**Monitoring**:
- Sentry - Error tracking
- Fail2ban - Intrusion prevention
- Logwatch - Log analyzer

### Security Resources

**Standards & Guides**:
- OWASP Top 10
- CWE (Common Weakness Enumeration)
- NIST Cybersecurity Framework

**Python Security**:
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [OWASP Python Security](https://owasp.org/www-project-python-security/)

**FastAPI Security**:
- [FastAPI Security Docs](https://fastapi.tiangolo.com/tutorial/security/)

---

## Contact

**Security Issues**: [security@nodo33.com] (or project maintainer)
**General Issues**: GitHub Issues
**Discussion**: GitHub Discussions

---

## Philosophy

> **"Proteggere senza controllare"** - Protect without controlling

Security in Nodo33 Sasso Digitale is about:
- **Protecting users** without invading privacy
- **Preventing harm** without restricting freedom
- **Being transparent** about what we protect and how
- **Failing safely** when systems are compromised

**Security serves the light. The light is not sold - it is gifted.**

---

**Sigillo**: 644
**Frequenza**: 300 Hz
**Motto**: La luce non si vende. La si regala.

**Fiat Amor, Fiat Risus, Fiat Lux** ✨

---

*Last updated: 2025-11-21*
*Version: 1.0.0*
