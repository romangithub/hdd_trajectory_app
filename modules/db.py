from __future__ import annotations
import sqlite3
from pathlib import Path
from datetime import datetime
import json

DB_FILE = Path(__file__).parent.parent / "projects.db"

def _connect():
    return sqlite3.connect(DB_FILE)

def init_db():
    conn = _connect()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS projects(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT,
      type TEXT,
      created_at TEXT,
      params TEXT,
      segments TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_project(name: str, type_: str, params: dict, segments=None):
    init_db()
    conn = _connect()
    conn.execute(
        "INSERT INTO projects (name, type, created_at, params, segments) VALUES (?, ?, ?, ?, ?)",
        (name, type_, datetime.now().isoformat(),
         json.dumps(params, ensure_ascii=False),
         json.dumps(segments, ensure_ascii=False) if segments is not None else None)
    )
    conn.commit()
    conn.close()

def list_projects(limit: int = 50):
    init_db()
    conn = _connect()
    cur = conn.execute("SELECT id, name, type, created_at FROM projects ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows

def load_project(project_id: int):
    conn = _connect()
    cur = conn.execute("SELECT id, name, type, created_at, params, segments FROM projects WHERE id=?", (project_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    id_, name, type_, created_at, params, segments = row
    return {
        "id": id_,
        "name": name,
        "type": type_,
        "created_at": created_at,
        "params": json.loads(params) if params else None,
        "segments": json.loads(segments) if segments else None
    }
