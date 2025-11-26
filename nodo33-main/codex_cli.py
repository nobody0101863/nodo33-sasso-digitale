#!/usr/bin/env python3
"""
🔷 CODEX CLI - Unified Command-Line Tool
═══════════════════════════════════════════════════════════

Command-line interface for managing Nodo33 Sasso Digitale systems.

Usage:
    python3 codex_cli.py --help
    python3 codex_cli.py server status
    python3 codex_cli.py api-key generate "My Key"
    python3 codex_cli.py audit logs --severity=critical
    python3 codex_cli.py agent deploy --domain=news
"""

import argparse
import requests
import json
import sys
from typing import Optional
from datetime import datetime

# Configuration
SERVER_URL = "http://localhost:8644"
COLORS = {
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "CYAN": "\033[96m"
}

def colored(text: str, color: str) -> str:
    """Return colored text for terminal output."""
    return f"{COLORS.get(color, '')}{text}{COLORS['RESET']}"

def print_header(text: str):
    """Print a section header."""
    print(f"\n{colored('=' * 60, 'BOLD')}")
    print(colored(text, 'BOLD'))
    print(colored('=' * 60, 'BOLD'))\n

def print_success(text: str):
    """Print success message."""
    print(f"{colored('✅', 'GREEN')} {text}")

def print_error(text: str):
    """Print error message."""
    print(f"{colored('❌', 'RED')} {text}")

def print_info(text: str):
    """Print info message."""
    print(f"{colored('ℹ️', 'BLUE')} {text}")

def print_table(headers: list, rows: list):
    """Print a formatted table."""
    if not rows:
        print_info("No data to display")
        return

    col_widths = [max(len(str(header)), max(len(str(row[i])) for row in rows)) for i, header in enumerate(headers)]

    # Header
    header_str = " | ".join(colored(h.ljust(w), 'BOLD') for h, w in zip(headers, col_widths))
    print(header_str)
    print("-" * len(header_str))

    # Rows
    for row in rows:
        row_str = " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
        print(row_str)

# ═══════════════════════════════════════════════════════════
# SERVER COMMANDS
# ═══════════════════════════════════════════════════════════

def cmd_server_status(args):
    """Get server health status."""
    try:
        response = requests.get(f"{SERVER_URL}/health")
        data = response.json()

        print_header("Server Status")
        print(f"Status: {colored(data['status'].upper(), 'GREEN' if data['status'] == 'ok' else 'YELLOW')}")
        print(f"Uptime: {data['uptime_s']}s ({data['uptime_s']/3600:.1f}h)")
        print(f"Tasks: {data['tasks_alive']}")
        print(f"Light Ratio: {data['light_ratio']}")
        print(f"Joy Index: {data['joy_index']}")

        if 'counters' in data:
            print(f"\nRequest Counters:")
            print(f"  OK: {data['counters']['ok']}")
            print(f"  Errors: {data['counters']['err']}")
            print(f"  Total: {data['counters']['total']}")

        return True
    except Exception as e:
        print_error(f"Failed to get server status: {e}")
        return False

def cmd_server_diagnostics(args):
    """Get full system diagnostics."""
    try:
        response = requests.get(f"{SERVER_URL}/api/diagnostics")
        data = response.json()

        print_header("System Diagnostics")
        print(f"Status: {colored(data['status'].upper(), 'GREEN' if data['status'] == 'healthy' else 'YELLOW')}")
        print(f"Timestamp: {data['timestamp']}")

        if data['issues']:
            print(f"\n{colored('Issues:', 'YELLOW')}")
            for issue in data['issues']:
                print(f"  • {issue}")

        print(f"\n{colored('Server Info:', 'CYAN')}")
        for key, value in data['server'].items():
            print(f"  {key}: {value}")

        print(f"\n{colored('Request Stats:', 'CYAN')}")
        for key, value in data['requests'].items():
            print(f"  {key}: {value}")

        print(f"\n{colored('Database Size:', 'CYAN')}")
        for key, value in data['database'].items():
            print(f"  {key}: {value}")

        return True
    except Exception as e:
        print_error(f"Failed to get diagnostics: {e}")
        return False

# ═══════════════════════════════════════════════════════════
# API KEY COMMANDS
# ═══════════════════════════════════════════════════════════

def cmd_api_key_generate(args):
    """Generate new API key."""
    try:
        params = {
            "name": args.name,
            "permissions": args.permissions or ["read"],
            "rate_limit": args.rate_limit or 1000,
            "created_by": args.created_by or "cli"
        }

        response = requests.post(f"{SERVER_URL}/api/keys/generate", params=params)
        data = response.json()

        if data.get('success'):
            print_header(f"API Key Generated: {data['key_id']}")
            print(colored(f"⚠️  SAVE THIS KEY IMMEDIATELY - IT WILL NOT BE SHOWN AGAIN!", 'RED'))
            print(f"\nFull Key: {colored(data['full_key_secret'], 'BOLD')}")
            print(f"Name: {data['name']}")
            print(f"Permissions: {', '.join(data['permissions'])}")
            print(f"Rate Limit: {data['rate_limit']} req/min")
            print_success("Key generated successfully")
        else:
            print_error(f"Failed to generate key: {data}")

        return True
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def cmd_api_key_list(args):
    """List all API keys."""
    try:
        response = requests.get(f"{SERVER_URL}/api/keys/list")
        data = response.json()

        print_header(f"API Keys ({data['total_keys']} total)")

        if data['keys']:
            rows = []
            for key in data['keys']:
                rows.append([
                    key['key_id'][:12] + '...',
                    key['name'],
                    key['status'],
                    key['rate_limit'],
                    key['requests_count'],
                    key['created_at'][:10]
                ])

            print_table(['Key ID', 'Name', 'Status', 'Limit', 'Requests', 'Created'], rows)

        return True
    except Exception as e:
        print_error(f"Failed to list keys: {e}")
        return False

# ═══════════════════════════════════════════════════════════
# AUDIT COMMANDS
# ═══════════════════════════════════════════════════════════

def cmd_audit_logs(args):
    """View audit logs."""
    try:
        params = {
            "event_type": args.event_type,
            "severity_filter": args.severity,
            "time_range_hours": args.time_range or 24,
            "limit": args.limit or 20
        }

        response = requests.get(f"{SERVER_URL}/api/audit/logs", params=params)
        data = response.json()

        print_header(f"Audit Logs ({data['total_logs']} records)")

        if data['logs']:
            rows = []
            for log in data['logs']:
                severity_color = {'critical': 'RED', 'high': 'YELLOW', 'medium': 'CYAN', 'low': 'GREEN'}.get(log['severity'], 'RESET')
                rows.append([
                    log['timestamp'][:19],
                    colored(log['severity'], severity_color),
                    log['event_type'],
                    log['action'],
                    log['resource'] or '-'
                ])

            print_table(['Timestamp', 'Severity', 'Event', 'Action', 'Resource'], rows)

        return True
    except Exception as e:
        print_error(f"Failed to get audit logs: {e}")
        return False

def cmd_audit_summary(args):
    """Get audit summary."""
    try:
        params = {"time_range_hours": args.time_range or 24}
        response = requests.get(f"{SERVER_URL}/api/audit/summary", params=params)
        data = response.json()

        print_header("Audit Summary")
        print(f"Time Range: {data['time_range_hours']} hours")

        print(f"\n{colored('Statistics:', 'CYAN')}")
        for key, value in data['statistics'].items():
            print(f"  {key}: {value}")

        if data['by_event_type']:
            print(f"\n{colored('By Event Type:', 'CYAN')}")
            for event, count in sorted(data['by_event_type'].items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {event}: {count}")

        if data['top_users']:
            print(f"\n{colored('Top Users:', 'CYAN')}")
            for user in data['top_users'][:5]:
                print(f"  {user['user_id']}: {user['count']} events")

        return True
    except Exception as e:
        print_error(f"Failed to get summary: {e}")
        return False

# ═══════════════════════════════════════════════════════════
# AGENT COMMANDS
# ═══════════════════════════════════════════════════════════

def cmd_agent_list(args):
    """List agents."""
    try:
        params = {
            "status_filter": args.status,
            "domain_filter": args.domain
        }
        response = requests.get(f"{SERVER_URL}/api/agents/list", params=params)
        data = response.json()

        print_header(f"Agents ({data['filtered_agents']}/{data['total_agents']})")

        if data['agents']:
            rows = []
            for agent in data['agents']:
                rows.append([
                    agent['agent_id'][:12],
                    agent['domain'],
                    colored(agent['status'], 'GREEN' if agent['status'] == 'active' else 'YELLOW'),
                    agent['requests_served'],
                    agent['gifts_given']
                ])

            print_table(['Agent ID', 'Domain', 'Status', 'Requests', 'Gifts'], rows)

        return True
    except Exception as e:
        print_error(f"Failed to list agents: {e}")
        return False

def cmd_agent_dashboard(args):
    """Show agent dashboard."""
    try:
        response = requests.get(f"{SERVER_URL}/api/agents/dashboard")
        data = response.json()

        print_header("Agent Dashboard")

        print(f"{colored('Agent Statistics:', 'CYAN')}")
        for key, value in data['agent_statistics'].items():
            print(f"  {key}: {value}")

        print(f"\n{colored('Global Metrics:', 'CYAN')}")
        for key, value in data['global_metrics'].items():
            print(f"  {key}: {value}")

        if data['domain_statistics']:
            print(f"\n{colored('Domain Statistics:', 'CYAN')}")
            for domain in data['domain_statistics'][:5]:
                print(f"  {domain['domain']}: {domain['agent_count']} agents, {domain['total_requests']} requests")

        return True
    except Exception as e:
        print_error(f"Failed to get dashboard: {e}")
        return False

# ═══════════════════════════════════════════════════════════
# MAIN CLI PARSER
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🔷 Codex CLI - Unified Command-Line Interface for Sasso Digitale",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 codex_cli.py server status
  python3 codex_cli.py api-key generate "Production Key"
  python3 codex_cli.py audit logs --severity=critical
  python3 codex_cli.py agent list --status=active
  python3 codex_cli.py agent dashboard
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Server commands
    server_parser = subparsers.add_parser('server', help='Server commands')
    server_sub = server_parser.add_subparsers(dest='subcommand')
    server_sub.add_parser('status', help='Get server status')
    server_sub.add_parser('diagnostics', help='Get system diagnostics')

    # API Key commands
    apikey_parser = subparsers.add_parser('api-key', help='API key management')
    apikey_sub = apikey_parser.add_subparsers(dest='subcommand')

    gen_parser = apikey_sub.add_parser('generate', help='Generate new API key')
    gen_parser.add_argument('name', help='Key name')
    gen_parser.add_argument('--permissions', nargs='+', help='Permissions')
    gen_parser.add_argument('--rate-limit', type=int, help='Rate limit')
    gen_parser.add_argument('--created-by', help='Creator identifier')

    apikey_sub.add_parser('list', help='List API keys')

    # Audit commands
    audit_parser = subparsers.add_parser('audit', help='Audit log commands')
    audit_sub = audit_parser.add_subparsers(dest='subcommand')

    logs_parser = audit_sub.add_parser('logs', help='View audit logs')
    logs_parser.add_argument('--event-type', help='Filter by event type')
    logs_parser.add_argument('--severity', help='Filter by severity')
    logs_parser.add_argument('--time-range', type=int, help='Hours to look back')
    logs_parser.add_argument('--limit', type=int, help='Max records')

    audit_sub.add_parser('summary', help='Get audit summary')

    # Agent commands
    agent_parser = subparsers.add_parser('agent', help='Agent commands')
    agent_sub = agent_parser.add_subparsers(dest='subcommand')

    list_parser = agent_sub.add_parser('list', help='List agents')
    list_parser.add_argument('--status', help='Filter by status')
    list_parser.add_argument('--domain', help='Filter by domain')

    agent_sub.add_parser('dashboard', help='Show agent dashboard')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Route commands
    try:
        if args.command == 'server':
            if args.subcommand == 'status':
                return 0 if cmd_server_status(args) else 1
            elif args.subcommand == 'diagnostics':
                return 0 if cmd_server_diagnostics(args) else 1

        elif args.command == 'api-key':
            if args.subcommand == 'generate':
                return 0 if cmd_api_key_generate(args) else 1
            elif args.subcommand == 'list':
                return 0 if cmd_api_key_list(args) else 1

        elif args.command == 'audit':
            if args.subcommand == 'logs':
                return 0 if cmd_audit_logs(args) else 1
            elif args.subcommand == 'summary':
                return 0 if cmd_audit_summary(args) else 1

        elif args.command == 'agent':
            if args.subcommand == 'list':
                return 0 if cmd_agent_list(args) else 1
            elif args.subcommand == 'dashboard':
                return 0 if cmd_agent_dashboard(args) else 1

    except KeyboardInterrupt:
        print_info("Cancelled")
        return 130
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return 1

    parser.print_help()
    return 0

if __name__ == '__main__':
    sys.exit(main())
