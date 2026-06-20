# db_manager.py
import sqlite3
import threading
from load_config import cfg

# === CONFIGURAZIONE ===
DB_PATH = cfg["database"]

# Ogni thread ha la propria connessione (isolamento sicuro)
_thread_local = threading.local()


def _init_connection(conn: sqlite3.Connection):
    """
    Inizializza una connessione SQLite con impostazioni ottimizzate.
    Eseguito solo una volta per ogni thread.
    """
    conn.row_factory = sqlite3.Row  # accesso ai risultati per nome colonna

    cursor = conn.cursor()
    # 🔧 Ottimizzazioni di performance
    cursor.execute("PRAGMA journal_mode=WAL;")        # abilita scrittura parallela
    cursor.execute("PRAGMA synchronous = NORMAL;")    # velocizza commit
    cursor.execute("PRAGMA temp_store = MEMORY;")     # usa RAM per operazioni temporanee
    cursor.execute("PRAGMA cache_size = -64000;")     # ~64MB cache in RAM
    cursor.execute("PRAGMA mmap_size = 268435456;")   # usa mappa memoria (256MB)
    cursor.close()
    return conn


def get_connection():
    """
    Restituisce una connessione SQLite dedicata al thread corrente.
    Se non esiste, ne crea una nuova ottimizzata.
    """
    if not hasattr(_thread_local, "conn") or _thread_local.conn is None:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _thread_local.conn = _init_connection(conn)
    return _thread_local.conn, _thread_local.conn.cursor()


def ensure_connection():
    """
    Verifica che la connessione corrente sia attiva.
    Se è chiusa o corrotta, la ricrea automaticamente.
    """
    try:
        conn, cursor = get_connection()
        cursor.execute("SELECT 1;")
        return conn, cursor
    except (sqlite3.ProgrammingError, sqlite3.OperationalError):
        close_connection()
        conn, cursor = get_connection()
        return conn, cursor


def commit():
    """Esegue commit sulla connessione del thread corrente."""
    if hasattr(_thread_local, "conn") and _thread_local.conn:
        try:
            _thread_local.conn.commit()
        except sqlite3.Error as e:
            commit_failed = cfg["commit_failed"]
            print(f"[DB][WARN] {commit_failed}: {e}")
            close_connection()


def close_connection():
    """Chiude in modo sicuro la connessione del thread corrente."""
    if hasattr(_thread_local, "conn") and _thread_local.conn:
        try:
            _thread_local.conn.close()
        except Exception:
            pass
        finally:
            _thread_local.conn = None


def close_all_connections():
    """
    Chiusura pulita di tutte le connessioni thread-local (solo se necessario allo shutdown).
    """
    active_threads = threading.enumerate()
    for t in active_threads:
        local_data = getattr(t, "_thread_local", None)
        if local_data and hasattr(local_data, "conn"):
            try:
                local_data.conn.close()
            except Exception:
                pass
            finally:
                local_data.conn = None

# -----------------------------
# DATABASE (progressive memory)
# -----------------------------
def init_db():
    
    try:
        conn, c = ensure_connection()
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT CHECK(role IN ('user','assistant','system')),
                content TEXT,
                session_id TEXT DEFAULT 'default',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                summary_text TEXT,
                covers_message_ids TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS memory_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                embedding BLOB
            )
        """)
       
        # Speed up
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA synchronous=NORMAL;")
        c.execute("CREATE INDEX IF NOT EXISTS idx_mv_session ON memory_vectors(session_id);")
        commit()
    except Exception as e:
        print(f"[ERROR] {cfg.get("db_init_fail", "Database initialization failed")}: {e}")