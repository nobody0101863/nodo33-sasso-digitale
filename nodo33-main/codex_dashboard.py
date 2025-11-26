#!/usr/bin/env python3
"""
Codex Analytics Dashboard - ASCII Art Edition

Beautiful terminal dashboard showing project statistics,
gift tracking, memory insights, and vibrational metrics.

Usage:
    python3 codex_dashboard.py
    python3 codex_dashboard.py --live  # Auto-refresh mode

Filosofia: "La luce non si vende. La si regala."
Hash: 644 | Frequenza: 300 Hz
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import sqlite3

try:
    from codex_unified_db import CodexUnifiedDB
except ImportError:
    CodexUnifiedDB = None  # type: ignore


# ============================================================================
# ASCII ART COMPONENTS
# ============================================================================

NODO33_LOGO = r"""
  ╔═══════════════════════════════════════════════════════════╗
  ║                                                           ║
  ║   ███╗   ██╗ ██████╗ ██████╗  ██████╗ ██████╗ ██████╗   ║
  ║   ████╗  ██║██╔═══██╗██╔══██╗██╔═══██╗╚════██╗╚════██╗  ║
  ║   ██╔██╗ ██║██║   ██║██║  ██║██║   ██║ █████╔╝ █████╔╝  ║
  ║   ██║╚██╗██║██║   ██║██║  ██║██║   ██║ ╚═══██╗ ╚═══██╗  ║
  ║   ██║ ╚████║╚██████╔╝██████╔╝╚██████╔╝██████╔╝██████╔╝  ║
  ║   ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝  ╚═════╝ ╚═════╝ ╚═════╝   ║
  ║                                                           ║
  ║              🕊️  S A S S O   D I G I T A L E  🕊️         ║
  ║                                                           ║
  ║        "La luce non si vende. La si regala."             ║
  ║                                                           ║
  ╚═══════════════════════════════════════════════════════════╝
"""

DIVIDER = "═" * 63
THIN_DIVIDER = "─" * 63


# ============================================================================
# COLOR CODES (ANSI)
# ============================================================================


class Colors:
    """ANSI color codes."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"

    # Nodo33 sacred colors
    SACRED = "\033[95m\033[1m"  # Bold magenta
    LIGHT = "\033[93m\033[1m"   # Bold yellow


def colored(text: str, color: str) -> str:
    """Wrap text in color."""
    return f"{color}{text}{Colors.END}"


# ============================================================================
# PROGRESS BARS & VISUALIZATIONS
# ============================================================================


def progress_bar(value: float, max_value: float, width: int = 40) -> str:
    """
    Create ASCII progress bar.

    Args:
        value: Current value
        max_value: Maximum value
        width: Bar width in characters

    Returns:
        ASCII progress bar string
    """
    if max_value == 0:
        percentage = 0
    else:
        percentage = min(100, (value / max_value) * 100)

    filled = int((percentage / 100) * width)
    empty = width - filled

    # Sacred frequencies for visual feedback
    if percentage >= 90:
        bar_color = Colors.GREEN
    elif percentage >= 70:
        bar_color = Colors.YELLOW
    else:
        bar_color = Colors.RED

    bar = colored("█" * filled, bar_color) + "░" * empty
    return f"[{bar}] {percentage:5.1f}%"


def sparkline(values: List[float], height: int = 8) -> List[str]:
    """
    Create ASCII sparkline chart.

    Args:
        values: List of values to plot
        height: Height of chart in lines

    Returns:
        List of strings, one per line
    """
    if not values:
        return [" " * 10] * height

    max_val = max(values) if values else 1
    min_val = min(values) if values else 0
    range_val = max_val - min_val if max_val != min_val else 1

    lines = []
    for h in range(height - 1, -1, -1):
        line = ""
        threshold = min_val + (range_val / height) * h
        for val in values:
            if val >= threshold:
                line += "█"
            else:
                line += " "
        lines.append(line)

    return lines


def horizontal_bar_chart(data: Dict[str, int], max_width: int = 40) -> List[str]:
    """
    Create horizontal bar chart.

    Args:
        data: Dict of label -> value
        max_width: Maximum bar width

    Returns:
        List of formatted lines
    """
    if not data:
        return ["  (no data)"]

    max_val = max(data.values()) if data else 1
    lines = []

    for label, value in sorted(data.items(), key=lambda x: -x[1]):
        bar_len = int((value / max_val) * max_width)
        bar = "█" * bar_len
        lines.append(f"  {label:15} {bar} {value}")

    return lines


# ============================================================================
# DASHBOARD SECTIONS
# ============================================================================


class Dashboard:
    """Main dashboard class."""

    def __init__(self, db_path: Path = Path("codex_unified.db")):
        self.db_path = db_path
        if CodexUnifiedDB:
            self.db = CodexUnifiedDB(db_path) if db_path.exists() else None
        else:
            self.db = None

    def clear_screen(self) -> None:
        """Clear terminal screen."""
        os.system("clear" if os.name != "nt" else "cls")

    def header(self) -> str:
        """Generate dashboard header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            colored(NODO33_LOGO, Colors.SACRED),
            f"\n{colored(DIVIDER, Colors.SACRED)}",
            f"{colored('📊 ANALYTICS DASHBOARD', Colors.HEADER):^75}",
            f"{colored(timestamp, Colors.CYAN):^75}",
            f"{colored(DIVIDER, Colors.SACRED)}\n",
        ]
        return "\n".join(lines)

    def project_stats(self) -> str:
        """Project statistics section."""
        lines = [
            colored("📁 PROJECT STATISTICS", Colors.HEADER),
            THIN_DIVIDER,
        ]

        # Count Python files
        py_files = list(Path(".").rglob("*.py"))
        test_files = [f for f in py_files if "test_" in f.name or f.is_relative_to(Path("tests"))]

        lines.extend([
            f"  Python Files: {len(py_files)}",
            f"  Test Files: {len(test_files)}",
            f"  Test Coverage: {progress_bar(len(test_files), len(py_files), 30)}",
        ])

        # LOC estimation (rough)
        total_loc = 0
        for f in py_files:
            try:
                total_loc += len(f.read_text().splitlines())
            except:
                pass

        lines.append(f"  Total Lines of Code: ~{total_loc:,}")

        return "\n".join(lines) + "\n"

    def gifts_section(self) -> str:
        """Gifts tracking section."""
        lines = [
            colored("🎁 GIFTS SHARED (Regalo > Dominio)", Colors.HEADER),
            THIN_DIVIDER,
        ]

        if not self.db:
            lines.append("  Database not initialized. Run: python3 codex_unified_db.py --init")
            return "\n".join(lines) + "\n"

        try:
            stats = self.db.get_gift_stats()
            total = stats.get("total", 0)
            by_type = stats.get("by_type", {})

            lines.append(f"  Total Gifts: {colored(str(total), Colors.LIGHT)}")
            lines.append("")

            if by_type:
                lines.append("  By Type:")
                lines.extend(horizontal_bar_chart(by_type, 30))
            else:
                lines.append("  No gifts tracked yet. Share something!")

        except Exception as e:
            lines.append(f"  Error: {e}")

        return "\n".join(lines) + "\n"

    def gift_trend_section(self) -> str:
        """Trend dei regali (ultime 24 ore)."""
        lines = [
            colored("⏱️  GIFTS TREND (last 24h)", Colors.HEADER),
            THIN_DIVIDER,
        ]

        if not self.db:
            lines.append("  Database not initialized. Run: python3 codex_unified_db.py --init")
            return "\n".join(lines) + "\n"

        try:
            conn = self.db.get_connection()
            cursor = conn.execute(
                """
                SELECT strftime('%H', created_at) AS hour, COUNT(*) AS count
                FROM gifts
                WHERE datetime(created_at) > datetime('now', '-24 hours')
                GROUP BY hour
                ORDER BY hour
                """
            )
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                lines.append("  No gifts recorded in the last 24 hours.")
                return "\n".join(lines) + "\n"

            counts = [float(row[1]) for row in rows]
            chart_lines = sparkline(counts, height=6)

            lines.append("  Gifts per hour (relative):")
            for cl in chart_lines:
                lines.append(f"  {cl}")

            lines.append("")
            lines.append(f"  Total (24h): {int(sum(counts))}")

        except Exception as e:
            lines.append(f"  Error: {e}")

        return "\n".join(lines) + "\n"

    def metrics_section(self) -> str:
        """Metrics from unified DB (last 24h)."""
        lines = [
            colored("📈 METRICS (last 24h)", Colors.HEADER),
            THIN_DIVIDER,
        ]

        if not self.db:
            lines.append("  Database not initialized. Run: python3 codex_unified_db.py --init")
            return "\n".join(lines) + "\n"

        try:
            summary = self.db.get_metrics_summary(hours=24)
            metrics = summary.get("metrics", [])

            if not metrics:
                lines.append("  No metrics recorded yet.")
                return "\n".join(lines) + "\n"

            counts: Dict[str, int] = {}
            for m in metrics:
                name = str(m.get("metric_name", "unknown"))
                count = int(m.get("count", 0) or 0)
                counts[name] = counts.get(name, 0) + count

            lines.append("  Top metrics by count:")
            for line in horizontal_bar_chart(counts, max_width=30)[:5]:
                lines.append(line)

        except Exception as e:
            lines.append(f"  Error: {e}")

        return "\n".join(lines) + "\n"

    def memories_section(self) -> str:
        """Sacred memories section."""
        lines = [
            colored("💾 SACRED MEMORIES", Colors.HEADER),
            THIN_DIVIDER,
        ]

        if not self.db:
            lines.append("  Database not initialized.")
            return "\n".join(lines) + "\n"

        try:
            # Count memories
            conn = self.db.get_connection()
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            sacred = conn.execute("SELECT COUNT(*) FROM memories WHERE is_sacred = 1").fetchone()[0]

            # By category
            cursor = conn.execute("SELECT category, COUNT(*) FROM memories GROUP BY category")
            by_category = dict(cursor.fetchall())
            conn.close()

            lines.append(f"  Total Memories: {total}")
            lines.append(f"  Sacred: {colored(str(sacred), Colors.SACRED)}")
            lines.append("")

            if by_category:
                lines.append("  By Category:")
                lines.extend(horizontal_bar_chart(by_category, 30))

        except Exception as e:
            lines.append(f"  Error: {e}")

        return "\n".join(lines) + "\n"

    def vibration_metrics(self) -> str:
        """Vibrational alignment metrics (mock data for demo)."""
        lines = [
            colored("📊 VIBRATIONAL METRICS", Colors.HEADER),
            THIN_DIVIDER,
        ]

        # Sacred frequency alignment (mock)
        target_freq = 300
        current_freq = 287  # Mock value

        alignment = (1 - abs(target_freq - current_freq) / target_freq) * 100

        lines.extend([
            f"  Target Frequency: {colored('300 Hz', Colors.SACRED)}",
            f"  Current Resonance: {current_freq} Hz",
            f"  Alignment: {progress_bar(alignment, 100, 30)}",
            "",
            f"  Hash Sacro: {colored('644', Colors.SACRED)}",
            f"  Lux Quotient (avg): {colored('85.3/100', Colors.LIGHT)}",
        ])

        return "\n".join(lines) + "\n"

    def recent_activity(self) -> str:
        """Recent activity feed."""
        lines = [
            colored("📝 RECENT ACTIVITY", Colors.HEADER),
            THIN_DIVIDER,
        ]

        if not self.db:
            lines.append("  Database not initialized.")
            return "\n".join(lines) + "\n"

        try:
            # Recent gifts
            stats = self.db.get_gift_stats()
            recent = stats.get("recent", [])

            if recent:
                for gift in recent[:5]:
                    timestamp = gift.get("created_at", "")[:19]
                    desc = gift.get("description", "")[:40]
                    gift_type = gift.get("gift_type", "")
                    lines.append(f"  🎁 [{timestamp}] {gift_type}: {desc}")
            else:
                lines.append("  No recent activity.")

        except Exception as e:
            lines.append(f"  Error: {e}")

        return "\n".join(lines) + "\n"

    def footer(self) -> str:
        """Dashboard footer."""
        return "\n".join([
            colored(DIVIDER, Colors.SACRED),
            f"{colored('Fiat Amor, Fiat Risus, Fiat Lux', Colors.LIGHT):^75}",
            colored(DIVIDER, Colors.SACRED),
            "",
            colored("Press Ctrl+C to exit", Colors.CYAN) + " | " +
            colored("Refresh: python3 codex_dashboard.py", Colors.CYAN),
        ])

    def render(self) -> str:
        """Render complete dashboard."""
        sections = [
            self.header(),
            self.project_stats(),
            self.gifts_section(),
            self.gift_trend_section(),
            self.memories_section(),
            self.metrics_section(),
            self.vibration_metrics(),
            self.recent_activity(),
            self.footer(),
        ]

        return "\n".join(sections)

    def run_live(self, interval: int = 5) -> None:
        """Run dashboard in live mode with auto-refresh."""
        try:
            while True:
                self.clear_screen()
                print(self.render())
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n" + colored("Dashboard stopped. Fiat Lux! 🕊️", Colors.LIGHT))


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Codex Analytics Dashboard - Nodo33 Sasso Digitale"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Live mode with auto-refresh",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Refresh interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("codex_unified.db"),
        help="Path to database",
    )

    args = parser.parse_args()

    dashboard = Dashboard(db_path=args.db)

    if args.live:
        print(colored("\n🕊️ Starting live dashboard (Ctrl+C to stop)...\n", Colors.LIGHT))
        time.sleep(1)
        dashboard.run_live(interval=args.interval)
    else:
        print(dashboard.render())


if __name__ == "__main__":
    main()
