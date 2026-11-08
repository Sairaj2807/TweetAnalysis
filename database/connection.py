import os
import sqlite3
from pathlib import Path

from config.settings import DATABASE_URL

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _sqlite_path_from_url(url: str) -> str:
    # expects sqlite:///relative/or/absolute/path.db
    return url.replace("sqlite:///", "", 1)


def get_connection() -> sqlite3.Connection:
    db_path = _sqlite_path_from_url(DATABASE_URL)
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    conn = get_connection()
    with open(SCHEMA_PATH, "r") as f:
        schema = f.read()
    conn.executescript(schema)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Initialized database at {DATABASE_URL}")
