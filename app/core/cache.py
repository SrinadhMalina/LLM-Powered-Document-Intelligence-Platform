import sqlite3
import hashlib
import json
from typing import Optional, Any

class ResponseCache:
    def __init__(self, db_path: str = "data/cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    query_hash TEXT PRIMARY KEY,
                    response TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _get_hash(self, query: str) -> str:
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()

    def get(self, query: str) -> Optional[Any]:
        query_hash = self._get_hash(query)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT response FROM cache WHERE query_hash = ?", (query_hash,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None

    def set(self, query: str, response: Any):
        query_hash = self._get_hash(query)
        response_json = json.dumps(response)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (query_hash, response) VALUES (?, ?)",
                (query_hash, response_json)
            )
            conn.commit()
