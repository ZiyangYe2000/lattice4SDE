"""Transactional, single-writer checkpoint storage."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = Path(str(self.path) + ".lock")
        self.lock = open(self.lock_path, "a+b")
        self.lock.seek(0)
        if os.name == "nt":
            import msvcrt
            if self.lock_path.stat().st_size == 0:
                self.lock.write(b"0")
                self.lock.flush()
                self.lock.seek(0)
            msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS records "
            "(kind TEXT, key TEXT, data TEXT, PRIMARY KEY(kind,key))"
        )
        self.db.commit()

    def put(self, kind: str, key: str, data: dict) -> bool:
        payload = json.dumps(data, allow_nan=False, sort_keys=True)
        with self.db:
            cursor = self.db.execute(
                "INSERT OR IGNORE INTO records VALUES (?,?,?)", (kind, key, payload)
            )
        return cursor.rowcount == 1

    def replace(self, kind: str, key: str, data: dict) -> None:
        payload = json.dumps(data, allow_nan=False, sort_keys=True)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO records VALUES (?,?,?)", (kind, key, payload))

    def has(self, kind: str, key: str) -> bool:
        return self.db.execute(
            "SELECT 1 FROM records WHERE kind=? AND key=?", (kind, key)
        ).fetchone() is not None

    def rows(self, kind: str) -> list[dict]:
        result = self.db.execute(
            "SELECT data FROM records WHERE kind=? ORDER BY key", (kind,)
        ).fetchall()
        return [json.loads(row[0]) for row in result]

    def count(self, kind: str) -> int:
        return int(self.db.execute(
            "SELECT COUNT(*) FROM records WHERE kind=?", (kind,)
        ).fetchone()[0])

    def close(self) -> None:
        self.db.close()
        self.lock.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

