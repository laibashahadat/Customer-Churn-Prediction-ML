import json
import sqlite3
from datetime import datetime

from config import DB_PATH


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            customer_input TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            predicted_probability REAL NOT NULL,
            risk_category TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_prediction_record(customer_input, predicted_class, predicted_probability, risk_category):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO prediction_history (created_at, customer_input, predicted_class, predicted_probability, risk_category)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(timespec="seconds"),
            json.dumps(customer_input, ensure_ascii=True),
            predicted_class,
            float(predicted_probability),
            risk_category,
        ),
    )
    conn.commit()
    conn_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    conn.close()
    return conn_id


def get_prediction_history(limit=50):
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT created_at, customer_input, predicted_class, predicted_probability, risk_category
        FROM prediction_history
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
