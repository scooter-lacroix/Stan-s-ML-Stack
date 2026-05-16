# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

Rusty Stack is in active development. Security updates are applied to the latest release.

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue in Rusty Stack, please report it responsibly.

### How to Report

**Email:** scooterlacroix@gmail.com

Please do **not** file public GitHub issues for security vulnerabilities.

### What to Include in Your Report

To help us address the issue quickly, please include:

1. **Description**: A clear description of the vulnerability
2. **Affected component**: Which part of Rusty Stack is affected (installer, TUI, specific component installer, telemetry, etc.)
3. **Reproduction steps**: Step-by-step instructions to reproduce the issue
4. **Impact**: What an attacker could achieve by exploiting this vulnerability
5. **Environment**: Your OS, ROCm version, GPU, and Rusty Stack version
6. **Proof of concept**: If applicable, a minimal example demonstrating the issue

### Response Timeline

| Timeframe | Action |
|-----------|--------|
| Within 48 hours | Acknowledge receipt of your report |
| Within 7 days | Initial assessment and severity classification |
| Within 30 days | Fix developed and tested |
| Upon fix release | Public disclosure coordination with reporter |

### Severity Classification

- **Critical**: Remote code execution, token/credential exposure, privilege escalation
- **High**: Data exfiltration, denial of service, significant information leaks
- **Medium**: Limited information disclosure, non-critical configuration manipulation
- **Low**: Minor information leaks, non-exploitable edge cases

### Scope

The following are in scope for our security policy:

- Rusty Stack source code (`rusty-stack/`)
- Installer scripts and bootstrapping logic
- SealedToken handling and credential management
- Telemetry data collection and submission
- Pre-built binary download and verification (SHA-256 checks)
- GitHub Actions workflows

The following are **out of scope**:

- Vulnerabilities in third-party dependencies (report to the upstream project)
- Issues in AMD ROCm itself (report to AMD)
- Social engineering attacks
- Denial of service via resource exhaustion on our GitHub-hosted infrastructure

### Disclosure Policy

- We request **90 days** from acknowledgment before public disclosure
- We will coordinate with the reporter on the disclosure timeline
- We will credit the reporter in the security advisory (unless anonymity is requested)

Thank you for helping keep Rusty Stack and our users safe.
