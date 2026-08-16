"""
backend/db.py — Analiz geçmişi için SQLite kalıcılık katmanı.

Her fonksiyon kendi kısa ömürlü bağlantısını açıp kapatır; route handler'lar
farklı threadpool worker thread'lerinde çalıştığından (bkz. backend/main.py)
paylaşılan bir sqlite3 bağlantısı thread-safe olmaz.
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

import config

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS analysis_history (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,
    filename            TEXT NOT NULL,
    predicted_class     TEXT NOT NULL,
    confidence          REAL NOT NULL,
    top2_class          TEXT NOT NULL,
    top2_probability    REAL NOT NULL,
    margin              REAL NOT NULL,
    status              TEXT NOT NULL,
    probabilities_json  TEXT NOT NULL
);
"""


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Veritabanı dosyasını ve tabloyu (yoksa) oluşturur."""
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    conn = _get_connection()
    try:
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()
    finally:
        conn.close()


def insert_analysis(
    filename: str,
    predicted_class: str,
    confidence: float,
    top2_class: str,
    top2_probability: float,
    margin: float,
    status: str,
    probabilities: List[Dict],
) -> Dict:
    """
    Bir analiz sonucunu veritabanına kaydeder.

    Returns:
        {'id': int, 'created_at': str} — oluşturulan satırın id'si ve zaman damgası.
    """
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO analysis_history
                (created_at, filename, predicted_class, confidence,
                 top2_class, top2_probability, margin, status, probabilities_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                filename,
                predicted_class,
                confidence,
                top2_class,
                top2_probability,
                margin,
                status,
                json.dumps(probabilities),
            ),
        )
        conn.commit()
        return {"id": cursor.lastrowid, "created_at": created_at}
    finally:
        conn.close()


def get_history(limit: Optional[int] = None) -> List[Dict]:
    """En yeni analiz en üstte olacak şekilde geçmiş kayıtlarını döndürür."""
    conn = _get_connection()
    try:
        query = "SELECT * FROM analysis_history ORDER BY id DESC"
        params = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def clear_history() -> int:
    """Tüm geçmiş kayıtlarını siler. Silinen satır sayısını döndürür."""
    conn = _get_connection()
    try:
        cursor = conn.execute("DELETE FROM analysis_history")
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()
