import os
import secrets
import sqlite3
from pathlib import Path

DB_PATH = Path(os.environ.get("SMARTDOC_DB_PATH", "smartdoc.db"))
SESSION_TTL_HOURS = 12


def get_connection():
    db_path = DB_PATH
    # Docker's short bind-mount syntax creates a directory when the host-side
    # database file does not exist yet.
    if db_path.is_dir():
        db_path = db_path / "smartdoc.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # permet d'accéder aux colonnes par nom
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            category TEXT NOT NULL,
            file_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            old_category TEXT NOT NULL,
            new_category TEXT NOT NULL,
            corrected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    session_columns = {
        column["name"] for column in cursor.execute("PRAGMA table_info(sessions)")
    }
    if "expires_at" not in session_columns:
        cursor.execute("ALTER TABLE sessions ADD COLUMN expires_at TIMESTAMP")
        cursor.execute(
            "UPDATE sessions SET expires_at = datetime(created_at, ?)",
            (f"+{SESSION_TTL_HOURS} hours",)
        )
    cursor.execute(
        "DELETE FROM sessions WHERE expires_at IS NULL OR expires_at <= CURRENT_TIMESTAMP"
    )

    conn.commit()
    conn.close()


def create_user(email, password_hash):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (email, password_hash) VALUES (?, ?)",
        (email, password_hash)
    )
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return user_id


def get_user_by_email(email):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    return user


def get_user_by_id(user_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return user


def create_session(user_id):
    """Crée un jeton de session qui expire après une durée limitée."""
    token = secrets.token_urlsafe(32)
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sessions (token, user_id, expires_at) "
        "VALUES (?, ?, datetime('now', ?))",
        (token, user_id, f"+{SESSION_TTL_HOURS} hours")
    )
    conn.commit()
    conn.close()
    return token


def get_session_user_id(token):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT user_id FROM sessions "
        "WHERE token = ? AND expires_at > CURRENT_TIMESTAMP",
        (token,)
    )
    row = cursor.fetchone()
    conn.close()
    return row["user_id"] if row else None


def rotate_session(token):
    """Replace a valid legacy token and return its replacement."""
    replacement = secrets.token_urlsafe(32)
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE sessions SET token = ? "
        "WHERE token = ? AND expires_at > CURRENT_TIMESTAMP",
        (replacement, token)
    )
    conn.commit()
    rotated = cursor.rowcount == 1
    conn.close()
    return replacement if rotated else None


def delete_session(token):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()


def add_document_record(user_id, filename, category, file_path):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO documents (user_id, filename, category, file_path) VALUES (?, ?, ?, ?)",
        (user_id, filename, category, file_path)
    )
    conn.commit()
    conn.close()


def get_user_documents(user_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents WHERE user_id = ?", (user_id,))
    docs = cursor.fetchall()
    conn.close()
    return docs


def log_correction(user_id, filename, old_category, new_category):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO corrections (user_id, filename, old_category, new_category) VALUES (?, ?, ?, ?)",
        (user_id, filename, old_category, new_category)
    )
    conn.commit()
    conn.close()