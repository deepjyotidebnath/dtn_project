"""SQLite persistence layer for the DTN simulator.

Stores simulation runs, per-run message records and per-run event logs so that
results can be inspected after the Streamlit session ends.
"""
import json
import math
import os
import sqlite3
from datetime import datetime

import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dtn_results.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT, label TEXT, protocol TEXT, seed TEXT,
    n_nodes INTEGER, steps INTEGER, comm_range REAL,
    total INTEGER, delivered INTEGER, pending INTEGER, expired INTEGER, dropped INTEGER,
    delivery_ratio REAL, avg_delay REAL, avg_hops REAL, transmissions INTEGER, params TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    run_id INTEGER, msg_id TEXT, source TEXT, destination TEXT, priority TEXT,
    ttl INTEGER, created_step INTEGER, status TEXT, delivered_step INTEGER,
    hops INTEGER, path TEXT
);
CREATE TABLE IF NOT EXISTS events (
    run_id INTEGER, step INTEGER, event_type TEXT, msg_id TEXT,
    from_node TEXT, to_node TEXT, ai_prob REAL, detail TEXT
);
"""


def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def _clean(v):
    """Convert NaN to None so SQLite stores NULL."""
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def save_run(label, protocol, params, metrics, messages, events):
    """Persist one run. Returns the new run_id."""
    init_db()
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO runs (created_at, label, protocol, seed, n_nodes, steps, comm_range,
               total, delivered, pending, expired, dropped, delivery_ratio, avg_delay,
               avg_hops, transmissions, params) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, protocol,
                str(params.get("seed")), params.get("n_nodes"), params.get("steps"),
                params.get("comm_range"), metrics["total"], metrics["delivered"],
                metrics["pending"], metrics["expired"], metrics["dropped"],
                _clean(metrics["delivery_ratio"]), _clean(metrics["avg_delay"]),
                _clean(metrics["avg_hops"]), metrics["transmissions"],
                json.dumps(params, default=str),
            ),
        )
        run_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (run_id, m["msg_id"], m["source"], m["destination"], m["priority"],
                 m["ttl"], m["created_step"], m["status"], m["delivered_step"],
                 m["hops"], m["path"])
                for m in messages
            ],
        )
        conn.executemany(
            "INSERT INTO events VALUES (?,?,?,?,?,?,?,?)",
            [
                (run_id, e["step"], e["type"], e["msg_id"], e["from"], e["to"],
                 _clean(e["prob"]), e["detail"])
                for e in events
            ],
        )
    return run_id


def load_runs():
    init_db()
    with get_conn() as conn:
        return pd.read_sql_query("SELECT * FROM runs ORDER BY run_id DESC", conn)


def load_messages(run_id):
    with get_conn() as conn:
        return pd.read_sql_query("SELECT * FROM messages WHERE run_id = ?", conn, params=(int(run_id),))


def load_events(run_id, limit=500):
    with get_conn() as conn:
        return pd.read_sql_query(
            "SELECT * FROM events WHERE run_id = ? ORDER BY step LIMIT ?",
            conn, params=(int(run_id), int(limit)),
        )


def clear_db():
    with get_conn() as conn:
        conn.executescript("DROP TABLE IF EXISTS runs; DROP TABLE IF EXISTS messages; DROP TABLE IF EXISTS events;")
    init_db()
