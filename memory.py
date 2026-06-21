"""
Memory management and context building for conversation history.
"""
import json
import numpy as np
import threading
from constants import SESSION_ID
from settings import runtime
from load_config import cfg
from utils import truncate
from db_manager import ensure_connection
from db_queries import get_all_summaries, get_recent_messages


def _load_vector_cache(vector_cache, vector_cache_lock, session_id=SESSION_ID, limit=None):
    """
    Load recent vectors for session_id into cache up to `limit`.
    Returns {"M": np.ndarray, "contents": [str,...]}.
    If already in cache, returns immediately.
    
    Args:
        vector_cache: Dictionary for caching vectors
        vector_cache_lock: Threading lock for safe access
        session_id: Session identifier
        limit: Maximum vectors to load (defaults to runtime.max_vectors_per_session)
        
    Returns:
        Dict with "M" (numpy array) and "contents" (list of text)
    """
    if limit is None:
        limit = runtime.max_vectors_per_session
        
    with vector_cache_lock:
        entry = vector_cache.get(session_id)
        if entry:
            return entry

    # Load from DB only if not in cache
    conn, cur = ensure_connection()
    cur.execute(
        "SELECT content, embedding FROM memory_vectors WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, limit)
    )
    rows = cur.fetchall()

    contents, mats = [], []
    for row in rows:
        content = row["content"]
        blob = row["embedding"]
        if blob:
            vec = np.frombuffer(blob, dtype=np.float32)
            mats.append(vec)
            contents.append(content)

    if mats:
        M = np.vstack(mats)
    else:
        M = np.zeros((0, 1), dtype=np.float32)

    new_entry = {"M": M, "contents": contents}
    with vector_cache_lock:
        vector_cache[session_id] = new_entry

    return new_entry

    
def vector_search(query, embed_text_func, vector_cache, vector_cache_lock, k=5, session_id=SESSION_ID):
    """
    Search vector memory for similar facts/messages.
    
    Args:
        query: Query text to search for
        embed_text_func: Function to embed text (from ollama_client)
        vector_cache: Dictionary for caching vectors
        vector_cache_lock: Threading lock for safe access
        k: Number of results to return
        session_id: Session identifier
        
    Returns:
        List of most similar text entries
    """
    q_emb = embed_text_func(query)
    if not q_emb:
        return []

    cache = _load_vector_cache(vector_cache, vector_cache_lock, session_id)
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


def build_context(session_id, query=None, embed_text_func=None, vector_cache=None, vector_cache_lock=None):
    """
    Build context messages suitable for Ollama chat.
    Includes: system identity, summaries, important facts, recent messages, 
    optional vector facts, and user query.
    
    Args:
        session_id: Session identifier
        query: Optional user query to include
        embed_text_func: Function to embed text (for vector search)
        vector_cache: Dictionary for caching vectors
        vector_cache_lock: Threading lock for safe access
        
    Returns:
        List of message dicts with "role" and "content"
    """
    messages = []

    # Context control parameters
    MAX_CONTEXT_TOKENS = 1200
    RESERVED_FOR_RESPONSE = 150
    SYSTEM_RESERVE = 300

    def token_estimate(text: str) -> int:
        """Approximate token count based on word count"""
        if not text:
            return 0
        return len(text.split())

    # 0) Fixed identities (always present) - keep but limit them
    identity_message = f"{cfg.get('identity_llm', 'Important note: I am Emma-Zira')}{cfg.get('identity_user', 'You are Marco.')}"
    identity_message = truncate(identity_message, limit=1000)
    messages.append({"role": "system", "content": identity_message})

    total_tokens = token_estimate(identity_message)

    # 1) Summaries -> single system message (insert but truncate)
    summaries = get_all_summaries(session_id, limit=runtime.max_summaries)
    if summaries:
        joined = "\n".join(summaries)
        summ_text = cfg.get("prev_summ", "Previous summaries:\n") + joined
        summ_text_trunc = truncate(summ_text, limit=4000)
        messages.append({"role": "system", "content": summ_text_trunc})
        total_tokens += token_estimate(summ_text_trunc)

    max_allowed = MAX_CONTEXT_TOKENS - RESERVED_FOR_RESPONSE
    if max_allowed < SYSTEM_RESERVE:
        max_allowed = MAX_CONTEXT_TOKENS - RESERVED_FOR_RESPONSE

    remaining_budget = max_allowed - total_tokens
    if remaining_budget < 0:
        remaining_budget = 0

    # 3) Recent verbatim messages: take the latest and insert while there is budget
    recent = get_recent_messages(session_id=session_id, limit=runtime.history_limit * 3)
    selected_recent = []
    for msg in reversed(recent):
        content = truncate(msg["content"], limit=1200)
        est = token_estimate(content)
        if est <= remaining_budget and remaining_budget > 0:
            selected_recent.insert(0, {"role": msg["role"], "content": content})
            remaining_budget -= est
        else:
            if remaining_budget > 10:
                words = content.split()
                take = max(5, remaining_budget)
                frag = " ".join(words[-take:])
                selected_recent.insert(0, {"role": msg["role"], "content": frag + "..."})
                remaining_budget = 0
            break

    messages.extend(selected_recent)
    total_tokens += token_estimate(" ".join(m["content"] for m in selected_recent))

    # 4) Vector facts for this query (optional) - add only if there is space
    if query and remaining_budget > 20 and embed_text_func and vector_cache is not None and vector_cache_lock is not None:
        facts = vector_search(query, embed_text_func, vector_cache, vector_cache_lock, k=5, session_id=session_id)
        if facts:
            facts_content = cfg.get("relev_mem_info", "Relevant information from memory:") + "\n" + "\n".join(facts)
            facts_content_trunc = truncate(facts_content, limit=1000)
            est = token_estimate(facts_content_trunc)
            if est <= remaining_budget:
                messages.append({"role": "system", "content": facts_content_trunc})
                remaining_budget -= est

    # 5) Finally add user query as last message (always)
    if query:
        user_query = truncate(query, limit=1000)
        messages.append({"role": "user", "content": user_query})
        total_tokens += token_estimate(user_query)

    # DEBUG: log if we truncated anything
    try:
        data = json.dumps(messages, ensure_ascii=False)
        if len(data) > 1000 or total_tokens > (MAX_CONTEXT_TOKENS * 0.8):
            print(f"[DEBUG][build_context]: total_tokens_est={total_tokens}, json_len={len(data)}")
    except Exception as e:
        from utils import Colors
        print(f"{Colors.ERROR}[ERROR] {cfg.get('payload_err', 'Payload error')} {e}")

    return messages
