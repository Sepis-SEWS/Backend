"""Shared SQLite schema and explicit transaction boundaries."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


def canonical(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


SCHEMA = """
CREATE TABLE IF NOT EXISTS encounters (
 id TEXT PRIMARY KEY, envelope TEXT NOT NULL, envelope_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
 encounter_id TEXT NOT NULL REFERENCES encounters(id), hour INTEGER NOT NULL,
 observed_at TEXT NOT NULL, raw TEXT NOT NULL, prefix_hash TEXT NOT NULL,
 PRIMARY KEY(encounter_id,hour)
);
CREATE TABLE IF NOT EXISTS predictions (
 id TEXT PRIMARY KEY, encounter_id TEXT NOT NULL, hour INTEGER NOT NULL,
 computed_at TEXT NOT NULL, imported_at TEXT NOT NULL, result TEXT NOT NULL,
 model_version TEXT NOT NULL, threshold_version TEXT NOT NULL,
 UNIQUE(encounter_id,hour),
 FOREIGN KEY(encounter_id,hour) REFERENCES observations(encounter_id,hour)
);
CREATE TABLE IF NOT EXISTS alerts (
 id TEXT PRIMARY KEY, encounter_id TEXT NOT NULL, hour INTEGER NOT NULL,
 snapshot TEXT NOT NULL, acknowledged_by TEXT, acknowledged_at TEXT,
 UNIQUE(encounter_id,hour),
 FOREIGN KEY(encounter_id,hour) REFERENCES observations(encounter_id,hour)
);
CREATE TABLE IF NOT EXISTS records (
 id TEXT PRIMARY KEY, alert_id TEXT NOT NULL REFERENCES alerts(id),
 actor TEXT NOT NULL, key TEXT NOT NULL, body TEXT NOT NULL, saved_at TEXT NOT NULL,
 UNIQUE(actor,alert_id,key)
);
"""


class Store:
    def __init__(self, path: Path):
        self.path = Path(path).resolve()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(SCHEMA)
        finally:
            connection.close()

    @contextmanager
    def connect(self, write: bool = False):
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        try:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
