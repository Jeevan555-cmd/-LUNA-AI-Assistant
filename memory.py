"""Persistent memory (SQLite). Survives restarts.

Tables:
  messages - the running conversation (short-term: the last few go to the LLM)
  profile  - lasting facts about you, one row per key (long-term)
  meta     - small bookkeeping values (used to notice when you edit profile.json)
"""
import json
import sqlite3
import threading
import time

import config

_conn = sqlite3.connect(config.MEMORY_DB, check_same_thread=False)
_lock = threading.Lock()   # several threads use the database, so they take turns

with _lock:
    _conn.execute(
        "CREATE TABLE IF NOT EXISTS messages ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, role TEXT, content TEXT)"
    )
    _conn.execute(
        "CREATE TABLE IF NOT EXISTS profile ("
        "key TEXT PRIMARY KEY, value TEXT NOT NULL, updated TEXT)"
    )
    _conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    _conn.commit()


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---------- conversation ----------
def save(role: str, content: str):
    with _lock:
        _conn.execute(
            "INSERT INTO messages (ts, role, content) VALUES (?, ?, ?)",
            (_now(), role, content),
        )
        _conn.commit()


def load_recent(n: int):
    """Last n messages, oldest first, in the format the LLM expects."""
    with _lock:
        rows = _conn.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


# ---------- profile (what Luna knows about you) ----------
def profile_load() -> dict:
    with _lock:
        rows = _conn.execute("SELECT key, value FROM profile").fetchall()
    return {k: json.loads(v) for k, v in rows}


def profile_replace(data: dict):
    """Make the profile table exactly equal to `data`."""
    with _lock:
        _conn.execute("DELETE FROM profile")
        _conn.executemany(
            "INSERT INTO profile (key, value, updated) VALUES (?, ?, ?)",
            [(k, json.dumps(v, ensure_ascii=False), _now()) for k, v in data.items()],
        )
        _conn.commit()


# ---------- bookkeeping ----------
def meta_get(key: str):
    with _lock:
        row = _conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def meta_set(key: str, value: str):
    with _lock:
        _conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
        _conn.commit()