"""
database/db_manager.py
--------------------------------------------------------------
Thread-safe SQLite database manager.

Tables
------
  faces   – one row per unique registered visitor
  events  – entry / exit events (timestamp, image path, confidence)
  stats   – running counters (unique_visitors, total_entries, total_exits)
"""

import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("face_tracker.db")


class DatabaseManager:
    """All database I/O in one place. Uses WAL mode for crash-resilience."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_schema()
        logger.info(f"[DB] Ready → {db_path}")

    # ── connection helper ────────────────────────────────────────────────
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ── schema ───────────────────────────────────────────────────────────
    def _init_schema(self):
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS faces (
                    id           TEXT PRIMARY KEY,
                    first_seen   TEXT NOT NULL,
                    last_seen    TEXT NOT NULL,
                    visit_count  INTEGER DEFAULT 1,
                    embedding    BLOB NOT NULL,
                    thumbnail    TEXT
                );

                CREATE TABLE IF NOT EXISTS events (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id      TEXT NOT NULL,
                    event_type   TEXT NOT NULL CHECK(event_type IN ('entry','exit')),
                    timestamp    TEXT NOT NULL,
                    image_path   TEXT,
                    frame_number INTEGER DEFAULT 0,
                    confidence   REAL    DEFAULT 1.0,
                    FOREIGN KEY (face_id) REFERENCES faces(id)
                );

                CREATE TABLE IF NOT EXISTS stats (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                INSERT OR IGNORE INTO stats(key, value) VALUES
                    ('unique_visitors', '0'),
                    ('total_entries',   '0'),
                    ('total_exits',     '0'),
                    ('session_start',   datetime('now'));
            """)

    # ── faces ─────────────────────────────────────────────────────────────
    def register_face(self, face_id: str, embedding: np.ndarray,
                      thumbnail: Optional[str] = None) -> bool:
        """Insert a new face record. Returns True on success, False if duplicate."""
        now  = datetime.now().isoformat()
        blob = embedding.astype(np.float32).tobytes()
        try:
            with self._conn() as c:
                c.execute(
                    "INSERT INTO faces(id, first_seen, last_seen, embedding, thumbnail)"
                    " VALUES(?, ?, ?, ?, ?)",
                    (face_id, now, now, blob, thumbnail)
                )
                c.execute(
                    "UPDATE stats SET value = CAST(CAST(value AS INT)+1 AS TEXT)"
                    " WHERE key = 'unique_visitors'"
                )
            logger.info(f"[DB] Registered: {face_id}")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"[DB] Already exists: {face_id}")
            return False

    def update_last_seen(self, face_id: str):
        now = datetime.now().isoformat()
        with self._conn() as c:
            c.execute(
                "UPDATE faces SET last_seen=?, visit_count=visit_count+1 WHERE id=?",
                (now, face_id)
            )

    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Load every stored embedding into memory for fast cosine matching."""
        with self._conn() as c:
            rows = c.execute("SELECT id, embedding FROM faces").fetchall()
        return [
            {
                "face_id":   r["id"],
                "embedding": np.frombuffer(r["embedding"], dtype=np.float32).copy()
            }
            for r in rows
        ]

    def get_all_faces(self) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, first_seen, last_seen, visit_count, thumbnail"
                " FROM faces ORDER BY first_seen DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_unique_visitor_count(self) -> int:
        with self._conn() as c:
            row = c.execute(
                "SELECT value FROM stats WHERE key='unique_visitors'"
            ).fetchone()
        return int(row["value"]) if row else 0

    # ── events ────────────────────────────────────────────────────────────
    def log_event(self, face_id: str, event_type: str,
                  image_path: Optional[str] = None,
                  frame_number: int = 0,
                  confidence: float = 1.0) -> int:
        """Insert an entry or exit event. Returns the new event id."""
        now      = datetime.now().isoformat()
        stat_key = "total_entries" if event_type == "entry" else "total_exits"
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO events(face_id, event_type, timestamp,"
                " image_path, frame_number, confidence)"
                " VALUES(?, ?, ?, ?, ?, ?)",
                (face_id, event_type, now, image_path, frame_number, confidence)
            )
            c.execute(
                f"UPDATE stats SET value = CAST(CAST(value AS INT)+1 AS TEXT)"
                f" WHERE key = '{stat_key}'"
            )
        return cur.lastrowid

    def get_recent_events(self, limit: int = 50) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, face_id, event_type, timestamp, image_path, confidence"
                " FROM events ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── stats ─────────────────────────────────────────────────────────────
    def get_stats(self) -> Dict[str, str]:
        with self._conn() as c:
            rows = c.execute("SELECT key, value FROM stats").fetchall()
        return {r["key"]: r["value"] for r in rows}
