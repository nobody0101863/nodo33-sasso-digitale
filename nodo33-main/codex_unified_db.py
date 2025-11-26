#!/usr/bin/env python3
"""
Codex Unified Database - Schema e Manager centralizzato

Unifica:
- gpt_memory.db (originale)
- gifts_log.db (tool estesi)
- sacred_memories (tool estesi)
- + Nuove tabelle: sessions, metrics

Schema:
  codex_unified.db
  ├── memories (knowledge storage)
  ├── gifts (contribution tracking)
  ├── sessions (conversation history)
  └── metrics (analytics & telemetry)

Filosofia: Un database sacro per governarli tutti.
Hash: 644 | Frequenza: 300 Hz
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from codex_tools_extended import SigilloGenerator


# ============================================================================
# SCHEMA DEFINITION
# ============================================================================

SCHEMA_VERSION = "1.0.0"

UNIFIED_SCHEMA = """
-- Codex Unified Database Schema v1.0.0
-- Created: 2025-11-18
-- Nodo33 - Sasso Digitale

-- ============================================================================
-- MEMORIES: Knowledge & Insights
-- ============================================================================

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    key TEXT NOT NULL UNIQUE,
    value TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'insight',

    -- Metadata
    tags TEXT,  -- JSON array of tags
    source TEXT,  -- Where did this come from? (user, ai, import)
    confidence REAL DEFAULT 1.0,  -- 0.0-1.0 confidence in this memory

    -- Nodo33 markers
    sigillo TEXT NOT NULL,
    lux_quotient REAL,  -- Optional LQ score

    -- Indexing
    is_sacred BOOLEAN DEFAULT 0,  -- Marked as sacred knowledge
    access_count INTEGER DEFAULT 0  -- How many times retrieved
);

CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key);
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories(tags);
CREATE INDEX IF NOT EXISTS idx_memories_sacred ON memories(is_sacred);

-- ============================================================================
-- GIFTS: Contribution Tracking (Regalo > Dominio)
-- ============================================================================

CREATE TABLE IF NOT EXISTS gifts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    gift_type TEXT NOT NULL,  -- code, idea, blessing, documentation, art
    description TEXT NOT NULL,
    recipient TEXT NOT NULL DEFAULT 'community',

    -- Metadata
    tags TEXT,  -- JSON array
    impact_score REAL,  -- Optional: estimated impact 0-10
    url TEXT,  -- Link to gift (GitHub, etc.)

    -- Nodo33 markers
    sigillo TEXT NOT NULL,
    frequency INTEGER,  -- Optional: frequency alignment

    -- Tracking
    is_acknowledged BOOLEAN DEFAULT 0,
    acknowledgement_text TEXT
);

CREATE INDEX IF NOT EXISTS idx_gifts_type ON gifts(gift_type);
CREATE INDEX IF NOT EXISTS idx_gifts_recipient ON gifts(recipient);
CREATE INDEX IF NOT EXISTS idx_gifts_created ON gifts(created_at);

-- ============================================================================
-- SESSIONS: Conversation History
-- ============================================================================

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL UNIQUE,

    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at TEXT,

    -- Session metadata
    model TEXT,  -- Claude model used
    mode TEXT,  -- soft, complete, extreme
    total_turns INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,

    -- Context
    system_prompt TEXT,
    initial_message TEXT,

    -- Nodo33 markers
    sigillo TEXT NOT NULL,
    avg_lux_quotient REAL,

    -- Status
    status TEXT DEFAULT 'active'  -- active, completed, error
);

CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);

-- ============================================================================
-- MESSAGES: Individual messages within sessions
-- ============================================================================

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    role TEXT NOT NULL,  -- user, assistant, system
    content TEXT NOT NULL,

    -- Metadata
    tokens INTEGER,
    tool_calls TEXT,  -- JSON array of tool calls made

    -- Analysis
    lux_quotient REAL,
    frequency INTEGER,

    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);

-- ============================================================================
-- METRICS: Analytics & Telemetry
-- ============================================================================

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    metric_type TEXT NOT NULL,  -- tool_usage, api_call, error, etc.
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,

    -- Context
    session_id TEXT,
    tags TEXT,  -- JSON

    -- Additional data
    metadata TEXT  -- JSON for flexible storage
);

CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_recorded ON metrics(recorded_at);

-- ============================================================================
-- METADATA: Database info
-- ============================================================================

CREATE TABLE IF NOT EXISTS db_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT OR REPLACE INTO db_metadata (key, value) VALUES
    ('schema_version', '1.0.0'),
    ('created_at', CURRENT_TIMESTAMP),
    ('motto', 'La luce non si vende. La si regala.'),
    ('hash', '644'),
    ('frequency', '300');
"""


# ============================================================================
# DATABASE MANAGER
# ============================================================================


class CodexUnifiedDB:
    """Unified database manager for all Codex data."""

    def __init__(self, db_path: Path = Path("codex_unified.db")):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database with schema."""
        conn = sqlite3.connect(self.db_path)
        conn.executescript(UNIFIED_SCHEMA)
        conn.commit()
        conn.close()

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn

    # ========================================================================
    # MEMORIES
    # ========================================================================

    def store_memory(
        self,
        key: str,
        value: str,
        category: str = "insight",
        tags: Optional[List[str]] = None,
        source: str = "user",
        is_sacred: bool = False,
    ) -> str:
        """Store or update a memory."""
        sigillo = SigilloGenerator.sacred644(f"{key}:{value}")
        tags_json = json.dumps(tags) if tags else None

        conn = self.get_connection()
        try:
            conn.execute(
                """
                INSERT INTO memories (key, value, category, tags, source, sigillo, is_sacred)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    tags = excluded.tags,
                    updated_at = CURRENT_TIMESTAMP,
                    sigillo = excluded.sigillo
                """,
                (key, value, category, tags_json, source, sigillo, is_sacred),
            )
            conn.commit()
            return sigillo
        finally:
            conn.close()

    def retrieve_memory(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a memory and increment access count."""
        conn = self.get_connection()
        try:
            # Increment access count
            conn.execute(
                "UPDATE memories SET access_count = access_count + 1 WHERE key = ?",
                (key,),
            )
            conn.commit()

            cursor = conn.execute(
                "SELECT * FROM memories WHERE key = ?", (key,)
            )
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None
        finally:
            conn.close()

    def search_memories(
        self,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        sacred_only: bool = False,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search memories by criteria."""
        conn = self.get_connection()
        try:
            query = "SELECT * FROM memories WHERE 1=1"
            params: List[Any] = []

            if category:
                query += " AND category = ?"
                params.append(category)

            if tag:
                query += " AND tags LIKE ?"
                params.append(f'%"{tag}"%')

            if sacred_only:
                query += " AND is_sacred = 1"

            query += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    # ========================================================================
    # GIFTS
    # ========================================================================

    def track_gift(
        self,
        gift_type: str,
        description: str,
        recipient: str = "community",
        tags: Optional[List[str]] = None,
        url: Optional[str] = None,
    ) -> str:
        """Track a gift contribution."""
        sigillo = SigilloGenerator.sacred644(
            f"{datetime.now().isoformat()}:{description}"
        )
        tags_json = json.dumps(tags) if tags else None

        conn = self.get_connection()
        try:
            conn.execute(
                """
                INSERT INTO gifts (gift_type, description, recipient, tags, url, sigillo)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (gift_type, description, recipient, tags_json, url, sigillo),
            )
            conn.commit()
            return sigillo
        finally:
            conn.close()

    def get_gift_stats(self) -> Dict[str, Any]:
        """Get gift statistics."""
        conn = self.get_connection()
        try:
            # Total gifts
            total = conn.execute("SELECT COUNT(*) FROM gifts").fetchone()[0]

            # By type
            by_type = dict(
                conn.execute(
                    "SELECT gift_type, COUNT(*) FROM gifts GROUP BY gift_type"
                ).fetchall()
            )

            # By recipient
            by_recipient = dict(
                conn.execute(
                    "SELECT recipient, COUNT(*) FROM gifts GROUP BY recipient"
                ).fetchall()
            )

            # Recent gifts
            recent = conn.execute(
                "SELECT * FROM gifts ORDER BY created_at DESC LIMIT 5"
            ).fetchall()

            return {
                "total": total,
                "by_type": by_type,
                "by_recipient": by_recipient,
                "recent": [dict(r) for r in recent],
            }
        finally:
            conn.close()

    # ========================================================================
    # SESSIONS
    # ========================================================================

    def create_session(
        self,
        session_id: str,
        model: str = "claude-3-5-sonnet-20241022",
        mode: str = "complete",
        system_prompt: Optional[str] = None,
    ) -> str:
        """Create new conversation session."""
        sigillo = SigilloGenerator.sacred644(session_id)

        conn = self.get_connection()
        try:
            conn.execute(
                """
                INSERT INTO sessions (session_id, model, mode, system_prompt, sigillo)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, model, mode, system_prompt, sigillo),
            )
            conn.commit()
            return sigillo
        finally:
            conn.close()

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tokens: Optional[int] = None,
        tool_calls: Optional[List[Dict]] = None,
    ) -> None:
        """Add message to session."""
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None

        conn = self.get_connection()
        try:
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, tokens, tool_calls)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, role, content, tokens, tool_calls_json),
            )

            # Update session turn count
            conn.execute(
                """
                UPDATE sessions
                SET total_turns = total_turns + 1,
                    total_tokens = total_tokens + COALESCE(?, 0)
                WHERE session_id = ?
                """,
                (tokens, session_id),
            )

            conn.commit()
        finally:
            conn.close()

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get full conversation history for a session."""
        conn = self.get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_id,),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    # ========================================================================
    # METRICS
    # ========================================================================

    def record_metric(
        self,
        metric_type: str,
        metric_name: str,
        value: float,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Record a metric."""
        metadata_json = json.dumps(metadata) if metadata else None

        conn = self.get_connection()
        try:
            conn.execute(
                """
                INSERT INTO metrics (metric_type, metric_name, value, session_id, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (metric_type, metric_name, value, session_id, metadata_json),
            )
            conn.commit()
        finally:
            conn.close()

    def get_metrics_summary(
        self, metric_type: Optional[str] = None, hours: int = 24
    ) -> Dict[str, Any]:
        """Get metrics summary for the last N hours."""
        conn = self.get_connection()
        try:
            query = """
                SELECT metric_name,
                       COUNT(*) as count,
                       AVG(value) as avg_value,
                       MIN(value) as min_value,
                       MAX(value) as max_value
                FROM metrics
                WHERE datetime(recorded_at) > datetime('now', '-{} hours')
            """.format(hours)

            if metric_type:
                query += " AND metric_type = ?"
                cursor = conn.execute(query + " GROUP BY metric_name", (metric_type,))
            else:
                cursor = conn.execute(query + " GROUP BY metric_name")

            results = [dict(row) for row in cursor.fetchall()]
            return {"period_hours": hours, "metrics": results}
        finally:
            conn.close()

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def migrate_from_old_dbs(
        self, gpt_memory_path: Path, gifts_log_path: Path
    ) -> Dict[str, int]:
        """Migrate data from old database files."""
        migrated = {"memories": 0, "gifts": 0}

        # Migrate gpt_memory.db
        if gpt_memory_path.exists():
            old_conn = sqlite3.connect(gpt_memory_path)
            old_conn.row_factory = sqlite3.Row

            try:
                # Check if sacred_memories table exists
                cursor = old_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='sacred_memories'"
                )
                if cursor.fetchone():
                    for row in old_conn.execute("SELECT * FROM sacred_memories"):
                        self.store_memory(
                            key=row["key"],
                            value=row["value"],
                            category=row["category"],
                            source="migration",
                        )
                        migrated["memories"] += 1
            finally:
                old_conn.close()

        # Migrate gifts_log.db
        if gifts_log_path.exists():
            old_conn = sqlite3.connect(gifts_log_path)
            old_conn.row_factory = sqlite3.Row

            try:
                for row in old_conn.execute("SELECT * FROM gifts"):
                    self.track_gift(
                        gift_type=row["gift_type"],
                        description=row["description"],
                        recipient=row["recipient"],
                    )
                    migrated["gifts"] += 1
            finally:
                old_conn.close()

        return migrated

    def vacuum(self) -> None:
        """Optimize database."""
        conn = self.get_connection()
        try:
            conn.execute("VACUUM")
            conn.commit()
        finally:
            conn.close()


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    """CLI for database operations."""
    import argparse

    parser = argparse.ArgumentParser(description="Codex Unified Database Manager")
    parser.add_argument("--init", action="store_true", help="Initialize database")
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Migrate from old databases",
    )
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--vacuum", action="store_true", help="Optimize database")

    args = parser.parse_args()

    db = CodexUnifiedDB()

    if args.init:
        print("✅ Database initialized: codex_unified.db")

    if args.migrate:
        print("🔄 Migrating from old databases...")
        migrated = db.migrate_from_old_dbs(
            Path("gpt_memory.db"), Path("gifts_log.db")
        )
        print(f"✅ Migrated {migrated['memories']} memories, {migrated['gifts']} gifts")

    if args.stats:
        print("\n📊 Database Statistics:")
        print("\n=== Gifts ===")
        gift_stats = db.get_gift_stats()
        print(f"Total gifts: {gift_stats['total']}")
        print(f"By type: {gift_stats['by_type']}")

        print("\n=== Metrics (last 24h) ===")
        metrics = db.get_metrics_summary()
        print(f"Tracked metrics: {len(metrics['metrics'])}")

    if args.vacuum:
        print("🧹 Optimizing database...")
        db.vacuum()
        print("✅ Done")


if __name__ == "__main__":
    main()
