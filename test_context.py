from load_config import load_configuration
from utils import clean_markdown, remove_emojis, truncate
from db_manager import ensure_connection, commit, close_all_connections
import json

loaded = load_configuration()   # Carica configurazione messaggi
cfg = loaded.get("cfg")

SESSION_ID = "default"
MAX_SUMMARIES = 15        # mid-term memory, always get the last n summaries
HISTORY_LIMIT = 20        # short term memory, keep last n turns verbatim

# Console colors
class Colors:
    USER = '\033[94m'
    PARTIAL = '\033[93m'
    ASSISTANT = '\033[92m'
    ERROR = '\033[91m'
    RESET = '\033[0m'
    
def get_recent_messages(session_id=SESSION_ID, limit=HISTORY_LIMIT):

    conn, c = ensure_connection()
    limit = int(limit)
    c.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, limit)
    )
    rows = c.fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]
    
def get_all_summaries(session_id=SESSION_ID, limit=MAX_SUMMARIES):

    conn, c = ensure_connection()
    c.execute("SELECT summary_text FROM summaries WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, limit))
    rows = [r[0] for r in c.fetchall()]
    return rows

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
    
output = build_context("default")
print(output)