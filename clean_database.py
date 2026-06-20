import contextlib
import sqlite3

MAX_VECTORS_PER_SESSION = 300
MAX_SUMMARIES = 30

def prune_old_vectors(session_id, db_path):

    @contextlib.contextmanager
    def get_db_connection():
        conn = sqlite3.connect(db_path)
        try:
            yield conn
        finally:
            conn.close()
        
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            # Count vectors for session
            c.execute("SELECT COUNT(*) FROM memory_vectors WHERE session_id=?", (session_id,))
            count = c.fetchone()[0]
            if count > MAX_VECTORS_PER_SESSION:
                # Delete oldest vectors
                excess = count - MAX_VECTORS_PER_SESSION
                c.execute("""
                    DELETE FROM memory_vectors 
                    WHERE id IN (
                        SELECT id FROM memory_vectors WHERE session_id=? ORDER BY id ASC LIMIT ?
                    )
                """, (session_id, excess))
                conn.commit()
                print(f"Cancellati {excess} record dalla tabella memory_vectors con session_id= {session_id}")
    except Exception as e:
        print(f"Error pruning old vectors for session {session_id}: {e}")

def prune_summaries(session_id, db_path):

    @contextlib.contextmanager
    def get_db_connection():
        conn = sqlite3.connect(db_path)
        try:
            yield conn
        finally:
            conn.close()
        
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            # Count vectors for session
            c.execute("SELECT COUNT(*) FROM summaries WHERE session_id=?", (session_id,))
            count = c.fetchone()[0]
            if count > MAX_SUMMARIES:
                # Delete oldest vectors
                excess = count - MAX_SUMMARIES
                c.execute("""
                    DELETE FROM summaries 
                    WHERE id IN (
                        SELECT id FROM summaries WHERE session_id=? ORDER BY id ASC LIMIT ?
                    )
                """, (session_id, excess))
                conn.commit()
                print(f"Cancellati {excess} record dalla tabella summaries con session_id= {session_id}")
    except Exception as e:
        print(f"Error pruning summaries for session {session_id}: {e}")        


prune_old_vectors('default','memory.db')
prune_summaries('default','memory.db')

prune_old_vectors('default','memory_english.db')
prune_summaries('default','memory_english.db')