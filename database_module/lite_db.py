"""
DECS - Standalone Embedded SQLite Database Module
=================================================
Provides asynchronous, zero-dependency persistence using SQLite in WAL mode.
Replaces heavy PostgreSQL / PostGIS requirement for low-spec deployments.
"""

import aiosqlite
import os

DB_PATH = "decs_local.db"


async def init_sqlite_db():
    """Initializes the local SQLite database and creates detection log table."""
    async with aiosqlite.connect(DB_PATH) as db:
        # Enable Write-Ahead Logging (WAL) for high concurrency and low latency
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA synchronous=NORMAL;")
        
        await db.execute("""
            CREATE TABLE IF NOT EXISTS detection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stream_id TEXT NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                engine TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()


async def save_detection_log(stream_id: str, class_name: str, confidence: float, lat: float, lon: float, engine: str = "ONNX_CPU"):
    """Inserts a single detection telemetry event into SQLite."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO detection_logs (stream_id, class_name, confidence, latitude, longitude, engine)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (stream_id, class_name, float(confidence), float(lat), float(lon), engine))
        await db.commit()


async def fetch_detection_history(min_confidence: float = 0.35, limit: int = 50):
    """Retrieves recent detection logs matching confidence threshold."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT id, stream_id, class_name, confidence, latitude, longitude, engine, timestamp
            FROM detection_logs
            WHERE confidence >= ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (min_confidence, limit)) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]