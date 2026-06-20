import requests
import json
from random import randrange
import pyaudio
import threading
import time
import re
import sqlite3
import numpy as np
import atexit
from load_config import load_configuration
from engine import TTS
from db_manager import ensure_connection, commit, close_all_connections
from queue import Queue, Empty
from faster_whisper import WhisperModel
import audioop
import piper
from piper import PiperVoice
import keyboard

# -----------------------------
# CONFIG
# -----------------------------
loaded = load_configuration()   # Carica configurazione messaggi
cfg = loaded.get("cfg")

LIBRARY = "qwen3:8b"
EMBED_MODEL = "nomic-embed-text"
SESSION_ID = "default"
SILENCE_THRESHOLD = 0.6   # sec of silence = end of phrase
HISTORY_LIMIT = 20        # short term memory, keep last n turns verbatim
SUMMARY_LIMIT = 50        # when > this, summarize older part
MAX_SUMMARIES = 15        # mid-term memory, always get the last n summaries
MAX_VECTORS_PER_SESSION = 200   #long term memory, keep the last n records
GOWORD = cfg.get("go_word")
STOPPHRASE = cfg.get("farewell")
VECTOR_CACHE = {} # --- cache ---
VECTOR_CACHE_LOCK = threading.Lock()
FACT_QUEUE = Queue()
VOICE_THRESHOLD = 400

# Console colors
class Colors:
    USER = '\033[94m'
    PARTIAL = '\033[93m'
    ASSISTANT = '\033[92m'
    ERROR = '\033[91m'
    RESET = '\033[0m'

# Lock per evitare chiamate TTS sovrapposte
tts_lock = threading.Lock()

# -----------------------------
# EVENT SYNCHRONIZATION (Sostituisce STATE legacy)
# -----------------------------
# L'evento controlla il flusso del microfono. 
# Quando è SET, il loop legge l'audio. Quando è CLEAR, il loop si mette in attesa (zero CPU).
assistant_speaking = threading.Event()

# -----------------------------
# WHISPER SETUP
# -----------------------------
whisper_model = WhisperModel(
    "small",
    device="cuda",
    compute_type="float16"
)

pa_condiviso = pyaudio.PyAudio()
try:
    stream = pa_condiviso.open(format=pyaudio.paInt16, channels=1, rate=16000,
                     input=True, frames_per_buffer=4000)
    stream.start_stream()
     
except Exception as e:
    print(f"{Colors.ERROR}❌ {cfg.get("mic_not_avail", "Error: Microphone not turned on or unavailable.")}")
    print(f"[ERROR] {cfg.get("err_details", "Details")}: {e}")
    pa_condiviso.terminate() # Pulisci prima di uscire
    exit(0)  # termina il programma in modo pulito

# Inizializzazione Piper TTS usando l'istanza PyAudio condivisa
piper_model = cfg.get("piper_model")
tts = TTS(piper_model, pa_instance=pa_condiviso)    

def transcribe_audio(frames):

    audio_bytes = b"".join(frames)

    audio = np.frombuffer(
        audio_bytes,
        np.int16
    ).astype(np.float32)

    audio /= 32768.0

    segments, info = whisper_model.transcribe(
        audio,
        language=cfg.get("user_lang"),
        vad_filter=True
    )

    text = "".join(
        segment.text 
        for segment in segments
    )

    return text.lower().strip()
# -----------------------------
# SPEAKTEXT (Aggiornata senza STATE)
# -----------------------------
def SpeakText(text: str, show_prompt=True):

    if not text.strip():
        return

    with tts_lock:

        assistant_speaking.set()

        print(
        f"\n{Colors.ASSISTANT}🔊 "
        f"{cfg.get('llm_speaking','Llm is speaking...')} "
        "[ESC per saltare]"
        f"{Colors.RESET}"
        )

        tts.start(text)

        while tts.is_speaking():

            if keyboard.is_pressed('esc'):
                tts.stop()
                break

            time.sleep(0.05)

        time.sleep(0.3)

        assistant_speaking.clear()

        if show_prompt:
            print(
            f"{Colors.USER}🎤 "
            f"{cfg.get('user_speaking','It is your turn')}"
            f"{Colors.RESET}\n"
            )
        
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
        c.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT,
                session_id TEXT
            )
        """)        
        # Speed up
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA synchronous=NORMAL;")
        c.execute("CREATE INDEX IF NOT EXISTS idx_mv_session ON memory_vectors(session_id);")
        commit()
    except Exception as e:
        print(f"[ERROR] {cfg.get("db_init_fail", "Database initialization failed")}: {e}")

# -----------------------------
# MEMORY WORKER
# -----------------------------
def memory_worker(batch_size=1, poll_interval=0.5):
    """
    Worker che consuma FACT_QUEUE, inserisce embedding in batch, invalidando cache,
    usando commit periodico.
    batch_size: quanti fatti inserire prima di un commit
    poll_interval: tempo di attesa quando la coda è vuota
    """
    from db_manager import ensure_connection

    buffer = []
    session_id = SESSION_ID  # puoi parametrizzare se supporti più sessioni

    while True:
        try:
            try:
                role, content, sid = FACT_QUEUE.get(timeout=poll_interval)
            except Empty:
                # se la coda è vuota, verifica se ci sono dati da flushare
                if buffer:
                    conn, cur = ensure_connection()
                    for (sid2, role2, cont2, blob2) in buffer:
                        cur.execute(
                            "INSERT INTO memory_vectors (session_id, role, content, embedding) VALUES (?,?,?,?)",
                            (sid2, role2, cont2, sqlite3.Binary(blob2))
                        )
                    conn.commit()
                    buffer.clear()
                    with VECTOR_CACHE_LOCK:
                        if session_id in VECTOR_CACHE:
                            del VECTOR_CACHE[session_id]
                continue  # torna all'inizio del ciclo

            emb = embed_text(content)
            if emb:
                vec = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    vec /= norm
                blob = vec.astype(np.float32).tobytes()
                buffer.append((sid, role, content, blob))

            # inserisci in batch se buffer pieno
            if len(buffer) >= batch_size:
                conn, cur = ensure_connection()
                for (sid2, role2, cont2, blob2) in buffer:
                    cur.execute(
                        "INSERT INTO memory_vectors (session_id, role, content, embedding) VALUES (?,?,?,?)",
                        (sid2, role2, cont2, sqlite3.Binary(blob2))
                    )
                conn.commit()
                buffer.clear()
                with VECTOR_CACHE_LOCK:
                    if session_id in VECTOR_CACHE:
                        del VECTOR_CACHE[session_id]

            FACT_QUEUE.task_done()  # ✅ chiamato solo se abbiamo effettivamente estratto un elemento

        except Exception as e:
            print(f"{Colors.ERROR}[memory_worker ERROR] {e}")
            time.sleep(poll_interval)

# -----------------------------
# VECTOR CACHE
# -----------------------------        
def _load_vector_cache(session_id=SESSION_ID, limit=MAX_VECTORS_PER_SESSION):
    """
    Carica in cache i vettori più recenti per session_id, fino a `limit`.  
    Ritorna {"M": np.ndarray, "contents": [str,...]}.
    Se già presente in cache, restituisce immediatamente.
    """
    with VECTOR_CACHE_LOCK:
        entry = VECTOR_CACHE.get(session_id)
        if entry:
            return entry

    # Carica dal DB solo se non in cache
    conn, cur = ensure_connection()
    # recupero embedding; assumo che embedding sia salvato come BLOB
    cur.execute(
        "SELECT content, embedding FROM memory_vectors WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, limit)
    )
    rows = cur.fetchall()  # con row_factory = sqlite3.Row

    contents, mats = [], []
    for row in rows:
        content = row["content"]
        blob = row["embedding"]
        if blob:
            vec = np.frombuffer(blob, dtype=np.float32)
            mats.append(vec)
            contents.append(content)

    if mats:
        M = np.vstack(mats)  # ogni riga normalizzata già
    else:
        M = np.zeros((0, 1), dtype=np.float32)

    new_entry = {"M": M, "contents": contents}
    with VECTOR_CACHE_LOCK:
        VECTOR_CACHE[session_id] = new_entry

    return new_entry
    
def add_message(role, content, session_id=SESSION_ID):

    try:
        conn, c = ensure_connection()
        c.execute("INSERT INTO messages (role, content, session_id) VALUES (?, ?, ?)", 
                  (role, content, session_id))
        commit()
    except Exception as e:
        print(f"{Colors.ERROR}[ERROR] {cfg.get("ins_msg_err", "Error writing to messages table.")}: {e}")

def get_recent_messages(session_id=SESSION_ID, limit=HISTORY_LIMIT):

    conn, c = ensure_connection()
    limit = int(limit)
    c.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, limit)
    )
    rows = c.fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]
        
def get_important_facts(session_id=SESSION_ID):

    conn, c = ensure_connection()
    c.execute(
        "SELECT fact FROM facts WHERE session_id=? ORDER BY id DESC",
        (session_id,)
    )
    rows = c.fetchall()
    return [row[0] for row in rows]       

def get_full_messages(session_id=SESSION_ID):

    conn, c = ensure_connection()
    c.execute("SELECT id, role, content FROM messages WHERE session_id=? ORDER BY id", (session_id,))
    rows = c.fetchall()
    return rows

def get_all_summaries(session_id=SESSION_ID, limit=MAX_SUMMARIES):

    conn, c = ensure_connection()
    c.execute("SELECT summary_text FROM summaries WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, limit))
    rows = [r[0] for r in c.fetchall()]
    return rows

def get_message_count(session_id):
    try:
        conn, c = ensure_connection()
        c.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
        result = c.fetchone()[0]
        return result
    except:
        return 0
# -----------------------------
# MEMORY HELPERS
# -----------------------------
def call_ollama(history):
    """
    Esegue una chiamata HTTP a Ollama /api/chat con gestione robusta degli errori,
    parsing rapido e fallback minimale in caso di risposte vuote.
    Ottimizzata per performance.
    """
    url = "http://127.0.0.1:11434/api/chat"
    payload = {
        "model": LIBRARY,
        "messages": history,
        "stream": False,
        "options": {
            "temperature": 0.3
        }
    }

    try:
        start = time.time()
        r = requests.post(url, json=payload, timeout=45)
        end = time.time()

        elapsed = end - start
        llm_resp_time = cfg.get("llm_resp_time", "LLM response")
        payload_chars = cfg.get("payload_chars", "chars")

        print(f"[DEBUG][call_ollama] {llm_resp_time}: {elapsed:.2f}s (payload {len(json.dumps(payload))} {payload_chars})")

        # parsing veloce, evita doppio .json() o errori silenziosi
        if r.status_code != 200:
            print(f"[WARN] Ollama HTTP {r.status_code}")
            return {"role": "assistant", "content": cfg.get("not_understood", "I did not understand.")}

        data = r.json()
        
        # log compatto per debug (solo in caso di risposta anomala)
        if not isinstance(data, dict) or "message" not in data:
            wait_oll_resp = cfg.get("wait_oll_resp", "Unexpected response from Ollama")
            print(f"[WARN] {wait_oll_resp}: {str(data)[:120]}...")
            return {"role": "assistant", "content": cfg.get("not_understood", "I did not understand.")}

        message = data.get("message", {})
        content = (message.get("content") or "").strip()

        if not content:
            empty_resp = cfg.get("empty_resp", "Empty response from llm model.")
            print(f"[WARN] {empty_resp}")
            content = cfg.get("not_understood", "I did not understand.")

        # ritorna solo il minimo necessario per il loop principale
        return {"role": "assistant", "content": content}

    except requests.Timeout:
        print(f"{Colors.ERROR}{cfg.get('llm_req_failed', 'Request timeout.')}")
        return {"role": "assistant", "content": cfg.get("llm_contact_failed", "Communication error with the model.")}

    except Exception as e:
        print(f"{Colors.ERROR}{cfg.get('llm_req_failed', 'Llm Request error')}: {e}{Colors.RESET}")
        return {"role": "assistant", "content": cfg.get("llm_contact_failed", "Communication error with the model.")}

def clean_markdown(text: str) -> str:
    text = re.sub(r'(\*{1,3})(.*?)\1', r'\2', text)  # bold/italic
    text = re.sub(r'#+\s*', '', text)                # headings
    text = text.replace('`', '')                     # code ticks
    text = re.sub(r'[-*_]{3,}', '', text)            # horizontal rulers
    text = re.sub(r'^\s*\*\s+', '', text, flags=re.MULTILINE) # rimozione bullet point (* inizio riga o dopo newline)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE) # elenco numerato (1. testo, 23. testo, ecc.)     
    return text.strip()

def truncate(text: str, limit: int = 500) -> str:
    """Tronca il testo se è troppo lungo, aggiungendo '...' alla fine."""
    return text[:limit] + "..." if len(text) > limit else text

def cleanup():
    """Chiusura sicura di risorse audio e database."""
    try:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
    except Exception:
        pass

    try:
        pa_condiviso.terminate()
    except Exception:
        pass

    try:
        close_all_connections()
        print(f"[DB] {cfg.get("conn_closed", "Connection closed automatically.")}")
    except Exception:
        pass

# 🔹 registra la funzione di cleanup per ogni tipo di uscita
atexit.register(cleanup)

# -----------------------------
# VECTOR MEMORY (FACT STORAGE)
# -----------------------------
def embed_text(text):
    url = "http://127.0.0.1:11434/api/embeddings"
    payload = {"model": EMBED_MODEL, "prompt": text}
    try:
        start = time.time()
        r = requests.post(url, json=payload, timeout=10)
        end = time.time()
        elapsed = end - start
        print(f"[DEBUG][embed_text] {cfg.get("embed_time", "Embedding time:")} {elapsed:.2f} sec ({cfg.get("txt_length", "text length")} {len(text)} {cfg.get("payload_chars", "chars")})")
        
        data = r.json()
        return data.get("embedding", [])
    except Exception as e:
        print(f"{Colors.ERROR}{cfg.get("embed_fail", "Embedding failed:")} {e}")
        return []

def vector_search(query, k=5, session_id=SESSION_ID):
    q_emb = embed_text(query)
    if not q_emb:
        return []

    cache = _load_vector_cache(session_id)
    M = cache["M"]
    if M.shape[0] == 0:
        return []

    q = np.array(q_emb, dtype=np.float32)
    q /= (np.linalg.norm(q) + 1e-9)

    sims = M @ q  # dot products
    if k >= len(sims):
        idx = np.argsort(-sims)
    else:
        idx = np.argpartition(-sims, k)[:k]
        idx = idx[np.argsort(-sims[idx])]

    return [cache["contents"][int(i)] for i in idx]

# -----------------------------
# BACKGROUND SUMMARIZER
# -----------------------------
def summarizer_worker(interval_sec=30):
    """
    Polls the DB periodically. If total messages for session > SUMMARY_LIMIT,
    it summarizes all but the last HISTORY_LIMIT messages into summaries,
    then deletes the summarized rows.
    """
    while True:
        try:
            rows = get_full_messages(SESSION_ID)  # (id, role, content)
            if len(rows) > SUMMARY_LIMIT:
                # everything except the most recent HISTORY_LIMIT turns
                to_summarize = rows[:-HISTORY_LIMIT]
                if to_summarize:
                    # summarize in chunks
                    chunk_size = 20
                    chunk_summaries = []
                    for i in range(0, len(to_summarize), chunk_size):
                        chunk = to_summarize[i:i+chunk_size]
                        text_block = "\n".join(f"{r}: {c}" for (_id, r, c) in chunk)

                        payload = {
                            "model": LIBRARY,
                            "messages": [
                                {"role": "system", "content": cfg.get("summarize_msg", "Summarize the following dialogue:")},
                                {"role": "user", "content": text_block}
                            ],
                            "stream": False
                        }
                        try:
                            resp = requests.post("http://127.0.0.1:11434/api/chat", json=payload, timeout=60)
                            resp.raise_for_status()
                            data = resp.json()
                            s = data.get("message", {}).get("content", "").strip()
                            if s:
                                chunk_summaries.append(s)
                        except Exception as e:
                            print(f"{Colors.ERROR}[Summarizer] {cfg.get("llm_call_failed", "Ollama call failed:")} {e}")
                            # continue with other chunks

                    if chunk_summaries:
                        final_summary = " ".join(chunk_summaries)
                        first_id = to_summarize[0][0]
                        last_id = to_summarize[-1][0]

                        conn, c = ensure_connection()
                        covers = f"{first_id}-{last_id}"
                        c.execute(
                            "INSERT INTO summaries (session_id, summary_text, covers_message_ids) VALUES (?,?,?)",
                            (SESSION_ID, final_summary, covers)
                        )

                        c.execute(
                            "DELETE FROM messages WHERE session_id=? AND id <= ?",
                            (SESSION_ID, last_id)
                        )
                        commit()                     

                        print(f"[Summarizer] {cfg.get("summ_saved", "Stored summary:")} ({len(to_summarize)} {cfg.get("turns_compr", "turns compressed")}).")
        except Exception as e:
            print(f"{Colors.ERROR}[SummarizerWorker] Error: {e}")
        finally:
            time.sleep(interval_sec)

# -----------------------------
# CONTEXT BUILDER
# -----------------------------
def build_context(session_id=SESSION_ID, query=None):
    """
    Return a list[ {role, content}, ... ] suitable for Ollama chat.
    Includes: system identity, summaries (single system msg), important facts,
    recent messages (troncati per token), optional vector facts, and the user query.
    Troncamento automatico del contesto per evitare prompt troppo lunghi.
    """
    messages = []

    # Parametri di controllo (aggiusta se necessario)
    MAX_CONTEXT_TOKENS = 1200       # limite stimato di "token" (approx = parole) per il prompt totale
    RESERVED_FOR_RESPONSE = 150     # lascia spazio per la risposta del modello
    SYSTEM_RESERVE = 300            # riserva minima di token per system messages (identità + summaries/facts)
    # Nota: questi valori sono empirici; aumentali se usi modelli più grandi.

    # helper: stima "token" approssimativa basata sulle parole
    def token_estimate(text: str) -> int:
        if not text:
            return 0
        return len(text.split())

    # 0) Fixed identities (sempre presenti) - manteniamo ma li limitiamo
    identity_message = f"{cfg.get("identity_llm", "Important note: I am Emma-Zira")}{cfg.get("identity_user", "You are Marco.")}"
    identity_message = truncate(identity_message, limit=1000)  # non troppo lunga
    messages.append({"role": "system", "content": identity_message})

    # inizializza contatore token con system identity
    total_tokens = token_estimate(identity_message)

    # 1) Summaries -> single system message (inserisco ma tronco)
    summaries = get_all_summaries(session_id)
    if summaries:
        joined = "\n".join(summaries)
        summ_text = cfg.get("prev_summ") + joined
        # tronca summaries più lunghe
        summ_text_trunc = truncate(summ_text, limit=4000)
        messages.append({"role": "system", "content": summ_text_trunc})
        total_tokens += token_estimate(summ_text_trunc)

    # 2) Important facts
    # important_facts = get_important_facts(session_id)
    # if important_facts:
        # joined_facts = "\n".join([f"- {fact}" for fact in important_facts])
        
        # user_facts = cfg.get("user_facts","Remember the following key facts about the user")
        # facts_text = f"{user_facts}:\n{joined_facts}"
        # facts_text_trunc = truncate(facts_text, limit=2000)
        # messages.append({"role": "system", "content": facts_text_trunc})
        # total_tokens += token_estimate(facts_text_trunc)

    # => se system messages già molto grandi, limitiamo ulteriormente il totale
    # calcoliamo il token budget rimanente per recent messages + vector facts + query
    max_allowed = MAX_CONTEXT_TOKENS - RESERVED_FOR_RESPONSE
    if max_allowed < SYSTEM_RESERVE:
        # garanzia minima
        max_allowed = MAX_CONTEXT_TOKENS - RESERVED_FOR_RESPONSE

    remaining_budget = max_allowed - total_tokens
    if remaining_budget < 0:
        remaining_budget = 0

    # 3) Recent verbatim messages: prendi gli ultimi e inseriscili finché c'è budget
    recent = get_recent_messages(session_id=session_id, limit=HISTORY_LIMIT * 3)  # prendi un poco più di history in caso
    # processa in reverse per raccogliere gli ultimi messaggi fino al budget
    selected_recent = []
    for msg in reversed(recent):  # partiamo dagli ultimi (più recenti)
        content = truncate(msg["content"], limit=1200)  # tronca singolo messaggio se troppo lungo
        est = token_estimate(content)
        if est <= remaining_budget and remaining_budget > 0:
            selected_recent.insert(0, {"role": msg["role"], "content": content})
            remaining_budget -= est
        else:
            # se non c'è spazio per l'intero messaggio, proviamo a inserire una versione più corta
            if remaining_budget > 10:
                # prova a inserire un frammento che si adatti
                words = content.split()
                take = max(5, remaining_budget)  # almeno qualche parola
                frag = " ".join(words[-take:])  # prendi la parte finale (più rilevante)
                selected_recent.insert(0, {"role": msg["role"], "content": frag + "..."})
                remaining_budget = 0
            break

    # aggiungi selected_recent in ordine cronologico corretto
    messages.extend(selected_recent)
    total_tokens += token_estimate(" ".join(m["content"] for m in selected_recent))

    # 4) Vector facts per questa query (opzionale) - aggiungi solo se c'è spazio
    if query and remaining_budget > 20:
        facts = vector_search(query, k=5, session_id=session_id)
        if facts:
            facts_content = cfg.get("relev_mem_info", "Relevant information from memory:") + "\n" + "\n".join(facts)
            facts_content_trunc = truncate(facts_content, limit=1000)
            est = token_estimate(facts_content_trunc)
            if est <= remaining_budget:
                messages.append({"role": "system", "content": facts_content_trunc})
                remaining_budget -= est
            else:
                # se non c'è spazio, ignora i vector facts (sono opzionali)
                pass

    # 5) Infine aggiungi la query utente come ultimo messaggio (sempre)
    if query:
        user_query = truncate(query, limit=1000)
        messages.append({"role": "user", "content": user_query})
        total_tokens += token_estimate(user_query)

    # DEBUG: se abbiamo troncato qualcosa, loggalo
    try:
        data = json.dumps(messages, ensure_ascii=False)

        # mostra la dimensione in caratteri e stima token
        if len(data) > 1000 or total_tokens > (MAX_CONTEXT_TOKENS * 0.8):
            print(f"[DEBUG][build_context]: total_tokens_est={total_tokens}, json_len={len(data)}")
    except Exception as e:
        print(f"{Colors.ERROR}[ERROR] {cfg.get("payload_err", "Payload error")} {e}")

    return messages

# -----------------------------
# COMMAND HELPERS
# -----------------------------
def runCommands(cmd, text):
    
    if cmd == cfg.get("commands")[0]:
        
        SpeakText(cfg.get("farewell", "goodbye"), show_prompt=False)
        stream.stop_stream()
        stream.close()
        pa_condiviso.terminate()
        close_all_connections()  # alla chiusura del programma
        exit(0)
    
    elif cmd == cfg.get("commands")[1]:
        print(f"{Colors.ASSISTANT}Assistant: {cfg.get("keyb_type_msg", "Okay, you can type from the keyboard.")}{Colors.RESET}")
        SpeakText(f"{cfg.get("keyb_type_msg", "Okay, you can type from the keyboard.")}", show_prompt=False)
        
        user_text = input(f"✍️ {cfg.get("write_here", "Write here:")} ").strip()
        if user_text:
            # Salva il testo scritto come messaggio utente
            add_message("user", user_text, SESSION_ID)
            FACT_QUEUE.put(("user", user_text, SESSION_ID))

        # Altrimenti, trattalo come input normale a Ollama
        history = build_context(session_id=SESSION_ID, query=user_text)
        response = call_ollama(history)
        assistant_text = response.get("content", "").strip()

        if assistant_text:
            assistant_text = clean_markdown(assistant_text)
            add_message("assistant", assistant_text, SESSION_ID)
            FACT_QUEUE.put(("assistant", assistant_text, SESSION_ID))

            print(f"{Colors.ASSISTANT}Assistant: {assistant_text}{Colors.RESET}")
            SpeakText(assistant_text, show_prompt=True)
    
def adaptive_memory_tuning(total_turns: int):
    """
    Adatta automaticamente la memoria del sistema in base alla durata della sessione.
    total_turns = numero totale di messaggi utente finora (es. count nella tabella messages)
    """
    global HISTORY_LIMIT, MAX_SUMMARIES, MAX_VECTORS_PER_SESSION
    # Livello 1️⃣ - Sessione breve
    if total_turns < 10:
        HISTORY_LIMIT = 20
        MAX_SUMMARIES = 5
        MAX_VECTORS_PER_SESSION = 100

    # Livello 2️⃣ - Sessione media
    elif total_turns < 30:
        HISTORY_LIMIT = 30
        MAX_SUMMARIES = 10
        MAX_VECTORS_PER_SESSION = 200

    # Livello 3️⃣ - Sessione lunga
    elif total_turns < 60:
        HISTORY_LIMIT = 40
        MAX_SUMMARIES = 15
        MAX_VECTORS_PER_SESSION = 300
        
    # Livello 4️⃣ - Sessione lunghissima (dialoghi di ore)
    else:
        HISTORY_LIMIT = 25   # accorcia per ridurre il rumore
        MAX_SUMMARIES = 20
        MAX_VECTORS_PER_SESSION = 400

# -----------------------------
# ASSISTANT LOOP
# -----------------------------
def assistant_loop():

    total_turns = get_message_count(SESSION_ID)
    adaptive_memory_tuning(total_turns)

    print(
        f"{Colors.ASSISTANT}Assistant: "
        f"{cfg.get('activ_welc_msg', 'System active. Waiting for activation word.')}"
        f"{Colors.RESET}"
    )

    SpeakText(
        cfg.get('activ_welc_msg',
        'System active. Waiting for activation word.'),
        show_prompt=True
    )

    audio_frames = []
    speech_started = False
    last_audio_time = time.time()

    try:

        while True:

            # Piper sta parlando, non ascoltare ma lasciare lo stream aperto
            if assistant_speaking.is_set():
                time.sleep(0.05)
                continue

            data = stream.read(4000, exception_on_overflow=False)
            audio_level = audioop.rms(data, 2)

            # -----------------------------
            # RILEVAMENTO VOCE
            # -----------------------------

            if audio_level > VOICE_THRESHOLD:

                # nuova frase, pulisco eventuale residuo precedente
                if not speech_started:
                    audio_frames.clear()

                speech_started = True

                # salvo anche il primo chunk
                audio_frames.append(data)
                last_audio_time = time.time()

            elif speech_started:

                # salvo il silenzio finale, utile per Whisper
                audio_frames.append(data)

                # fine frase
                if time.time() - last_audio_time > SILENCE_THRESHOLD:

                    final_text = transcribe_audio(audio_frames)

                    audio_frames.clear()
                    speech_started = False

                    if not final_text.strip():
                        continue

                    print(
                        f"{Colors.USER}🎤 User: "
                        f"{final_text}"
                        f"{Colors.RESET}"
                    )

                    # -----------------------------
                    # WAKE WORD
                    # -----------------------------

                    if GOWORD in final_text:

                        misc_array = cfg.get("acknowledgements")
                        SpeakText(misc_array[randrange(0, len(misc_array))])

                        continue

                    # -----------------------------
                    # STOP PHRASE
                    # -----------------------------

                    if STOPPHRASE in final_text:

                        misc_pts_array = cfg.get("part")
                        SpeakText(misc_pts_array[randrange(0, len(misc_pts_array))])

                        continue

                    # -----------------------------
                    # COMMANDI
                    # -----------------------------

                    executed = False
                    normalized = final_text.lower().strip()

                    command_trigger = cfg.get("command_trigger")

                    if normalized.startswith(command_trigger):

                        parts = normalized[len(command_trigger):].split(maxsplit=1)
                        inner_cmd = parts[0]

                        inner_text = (
                            parts[1]
                            if len(parts) > 1
                            else ""
                        )

                        cmd_array = cfg.get("commands")

                        if inner_cmd in cmd_array:

                            runCommands(inner_cmd, inner_text)
                            executed = True

                    if executed:
                        continue

                    # -----------------------------
                    # SALVA MESSAGGIO UTENTE
                    # -----------------------------

                    add_message("user", final_text, SESSION_ID)
                    FACT_QUEUE.put(("user", final_text, SESSION_ID))

                    # -----------------------------
                    # COSTRUZIONE CONTESTO
                    # -----------------------------
                    context_messages = build_context(session_id=SESSION_ID, query=final_text)

                    # -----------------------------
                    # OLLAMA
                    # -----------------------------

                    response = call_ollama(context_messages)
                    assistant_text = response.get("content", cfg.get("not_understood", "I did not understand."))
                    assistant_text = clean_markdown(assistant_text)

                    print(
                        f"{Colors.ASSISTANT}"
                        f"Assistant: {assistant_text}"
                        f"{Colors.RESET}"
                    )

                    # -----------------------------
                    # SALVA RISPOSTA prima del TTS
                    # -----------------------------

                    if assistant_text.strip():

                        add_message("assistant", assistant_text, SESSION_ID)
                        FACT_QUEUE.put(("assistant", assistant_text, SESSION_ID))
                        SpeakText(assistant_text)

    except Exception as e:

        print(f"{Colors.ERROR}[ERROR] {e}")
        
# -----------------------------
# MAIN LOOP
# -----------------------------
if __name__ == "__main__":
    init_db()
    threading.Thread(target=summarizer_worker, daemon=True).start()
    threading.Thread(target=memory_worker, daemon=True).start()
    
    try:
        assistant_loop()
    except KeyboardInterrupt:
        print(cfg.get("llm_stopped", "Assistant terminated."))
        print(f"\n[INFO] {cfg.get("keyb_interrupt", "Keyboard abort, closing.")}")
        # il cleanup verrà comunque eseguito automaticamente da atexit
