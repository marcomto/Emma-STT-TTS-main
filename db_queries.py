"""
Database queries module - funzioni di accesso ai dati.
Indipendente da thread e risorse audio condivise.
"""
from db_manager import ensure_connection, commit
from utils import Colors
from constants import SESSION_ID
from settings import runtime


def add_message(role, content, session_id=SESSION_ID):
    """Inserisce un messaggio nella tabella messages."""
    try:
        conn, c = ensure_connection()
        c.execute("INSERT INTO messages (role, content, session_id) VALUES (?, ?, ?)", 
                  (role, content, session_id))
        commit()
    except Exception as e:
        print(f"{Colors.ERROR}[ERROR] Error writing to messages table: {e}")


def get_recent_messages(session_id=SESSION_ID, limit=runtime.history_limit):
    """Recupera gli ultimi N messaggi per una sessione."""
    conn, c = ensure_connection()
    limit = int(limit)
    c.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, limit)
    )
    rows = c.fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def get_full_messages(session_id=SESSION_ID):
    """Recupera tutti i messaggi per una sessione."""
    conn, c = ensure_connection()
    c.execute("SELECT id, role, content FROM messages WHERE session_id=? ORDER BY id", (session_id,))
    rows = c.fetchall()
    return rows


def get_all_summaries(session_id=SESSION_ID, limit=runtime.max_summaries):
    """Recupera i sommari più recenti per una sessione."""
    conn, c = ensure_connection()
    c.execute("SELECT summary_text FROM summaries WHERE session_id=? ORDER BY id DESC LIMIT ?", 
              (session_id, limit))
    rows = [r[0] for r in c.fetchall()]
    return rows


def get_message_count(session_id):
    """Conta i messaggi per una sessione."""
    try:
        conn, c = ensure_connection()
        c.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
        result = c.fetchone()[0]
        return result
    except:
        return 0
