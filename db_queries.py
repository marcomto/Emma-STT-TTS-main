"""
Database queries module - data access functions.
Independent of threads and shared audio resources.
"""
from db_manager import ensure_connection, commit
from utils import Colors
from constants import SESSION_ID
from settings import runtime


def add_message(role, content, session_id=SESSION_ID):
    """Inserts a message into the messages table."""
    try:
        conn, c = ensure_connection()
        c.execute("INSERT INTO messages (role, content, session_id) VALUES (?, ?, ?)", 
                  (role, content, session_id))
        commit()
    except Exception as e:
        print(f"{Colors.ERROR}[ERROR] Error writing to messages table: {e}")


def get_recent_messages(session_id=SESSION_ID, limit=runtime.history_limit):
    """Retrieves the last N messages for a session."""
    conn, c = ensure_connection()
    limit = int(limit)
    c.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, limit)
    )
    rows = c.fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def get_full_messages(session_id=SESSION_ID):
    """Retrieves all messages for a session."""
    conn, c = ensure_connection()
    c.execute("SELECT id, role, content FROM messages WHERE session_id=? ORDER BY id", (session_id,))
    rows = c.fetchall()
    return rows


def get_all_summaries(session_id=SESSION_ID, limit=runtime.max_summaries):
    """Retrieves the most recent summaries for a session."""
    conn, c = ensure_connection()
    c.execute("SELECT summary_text FROM summaries WHERE session_id=? ORDER BY id DESC LIMIT ?", 
              (session_id, limit))
    rows = [r[0] for r in c.fetchall()]
    return rows


def get_message_count(session_id):
    """Retrieves the count of messages for a session."""
    try:
        conn, c = ensure_connection()
        c.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
        result = c.fetchone()[0]
        return result
    except:
        return 0
