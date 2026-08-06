"""
persistence.py
Logs entry/exit/occupancy snapshots to SQLite so counts survive a restart
and can be exported for daily reports.
"""

import sqlite3
import threading

import config


class Database:
    def __init__(self):
        self.conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS occupancy_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT (datetime('now','localtime')),
                    entry_count INTEGER,
                    exit_count INTEGER,
                    occupancy INTEGER
                )
            """)
            self.conn.commit()

    def log(self, entry_count, exit_count, occupancy):
        with self.lock:
            self.conn.execute(
                "INSERT INTO occupancy_log (entry_count, exit_count, occupancy) VALUES (?, ?, ?)",
                (entry_count, exit_count, occupancy)
            )
            self.conn.commit()
