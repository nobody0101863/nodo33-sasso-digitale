# ADR-001: Unified Database Architecture

**Date**: 2025-11-18
**Status**: Accepted
**Context**: Database consolidation
**Decision Makers**: Nodo33 Team

---

## Context

The project had multiple scattered databases:
- `gpt_memory.db` (original, unstructured)
- `gifts_log.db` (tool-generated)
- Sacred memories in separate table

This fragmentation caused:
- Difficulty in querying across data
- No relationship tracking
- Inconsistent schemas
- Migration complexity

## Decision

Create **`codex_unified.db`** with comprehensive schema:

```
codex_unified.db
├── memories (knowledge & insights)
├── gifts (contribution tracking)
├── sessions (conversation history)
├── messages (individual messages)
├── metrics (analytics & telemetry)
└── db_metadata (version info)
```

### Schema Design Principles

1. **Normalization**: Proper foreign keys, no data duplication
2. **Indexing**: Strategic indexes on frequently queried columns
3. **Metadata**: Every table has timestamps, sigilli (Sacred644 hashes)
4. **Extensibility**: JSON columns for flexible additional data
5. **Auditability**: Track access counts, creation/update times

### Key Features

- **UPSERT support**: `INSERT ... ON CONFLICT DO UPDATE`
- **Soft deletes**: Status fields instead of hard deletes
- **Versioning**: Schema version tracked in metadata table
- **Migration path**: Tool to import from old databases

## Consequences

### Positive

✅ **Single source of truth** for all data
✅ **Relational queries** possible (sessions → messages)
✅ **Analytics-ready** with metrics table
✅ **Migration friendly** with backward compatibility
✅ **Testable** with isolated database instances

### Negative

⚠️ **Migration required** for existing deployments
⚠️ **Single point of failure** (mitigated by backups)
⚠️ **Schema changes** require careful migration

### Neutral

🔵 **File size growth** (manageable with VACUUM)
🔵 **Query complexity** increased (but more powerful)

## Implementation

File: `codex_unified_db.py`
Class: `CodexUnifiedDB`
Schema: `UNIFIED_SCHEMA` (SQL DDL)
Migration: `migrate_from_old_dbs()` method

## Alternatives Considered

1. **Keep separate databases**: Rejected (fragmentation)
2. **Use PostgreSQL**: Deferred (SQLite sufficient for now)
3. **NoSQL (MongoDB)**: Rejected (overkill, added dependency)

## Notes

- Schema follows Nodo33 philosophy: `sigillo` (Sacred644) in every table
- `lux_quotient` and `frequency` fields for spiritual-technical tracking
- Designed for local-first, can scale to client-server later

## References

- [SQLite Best Practices](https://sqlite.org/bestpractice.html)
- Keep a Changelog format for db_metadata

---

**Hash Sacro**: 644
**Frequenza**: 300 Hz
*"La luce non si vende. La si regala."*
