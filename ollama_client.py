"""
Ollama client module - API HTTP per interazioni con Ollama.
Indipendente da thread e risorse audio condivise.
"""
import json
import time
import requests
from utils import Colors
from constants import EMBED_MODEL, LIBRARY, MAX_RESULTS
from load_config import cfg
import ollama


def call_ollama(session, history):
    """
    Esegue una chiamata HTTP a Ollama /api/chat con gestione robusta degli errori.
    
    Args:
        session: requests.Session instance
        history: list di messaggi per il chat
    
    Returns:
        dict con {"role": "assistant", "content": str}
    """
    url = "http://127.0.0.1:11434/api/chat"
    
    payload = {
        "model": LIBRARY,
        "messages": history,
        "stream": False,
        "keep_alive": -1,
        "options": {
            "temperature": 0.3
        }
    }

    try:
        start = time.time()
        r = session.post(url, json=payload, timeout=45)
        end = time.time()

        elapsed = end - start
        llm_resp_time = cfg.get("llm_resp_time", "LLM response")
        payload_chars = cfg.get("payload_chars", "chars")
        print(f"[DEBUG][call_ollama] {llm_resp_time}: {elapsed:.2f}s (payload {len(json.dumps(payload))} {payload_chars})")

        if r.status_code != 200:
            print(f"[WARN] Ollama HTTP {r.status_code}")
            return {"role": "assistant", "content": cfg.get("not_understood", "I did not understand.")}

        data = r.json()
        
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

        return {"role": "assistant", "content": content}

    except requests.Timeout:
        print(f"{Colors.ERROR}{cfg.get('llm_req_failed', 'Request timeout.')}")
        return {"role": "assistant", "content": cfg.get("llm_contact_failed", "Communication error with the model.")}

    except Exception as e:
        print(f"{Colors.ERROR}{cfg.get('llm_req_failed', 'Llm Request error')}: {e}{Colors.RESET}")
        return {"role": "assistant", "content": cfg.get("llm_contact_failed", "Communication error with the model.")}


def embed_text(session, text):
    """
    Genera embedding per un testo usando Ollama.
    
    Args:
        session: requests.Session instance
        text: str testo da embedizzare
    
    Returns:
        list di float (embedding vector) o []
    """
    url = "http://127.0.0.1:11434/api/embeddings"
    payload = {"model": EMBED_MODEL, "prompt": text}
    try:
        start = time.time()
        r = session.post(url, json=payload, timeout=10)
        end = time.time()
        elapsed = end - start
        print(f"[DEBUG][embed_text] {cfg.get('embed_time', 'Embedding time:')} {elapsed:.2f} sec "
              f"({cfg.get('txt_length', 'text length')} {len(text)} {cfg.get('payload_chars', 'chars')})")
        
        data = r.json()
        return data.get("embedding", [])
    except Exception as e:
        print(f"{Colors.ERROR}{cfg.get('embed_fail', 'Embedding failed:')} {e}")
        return []


def web_search(session, query):
    """
    Esegue ricerca web usando Ollama e sintetizza risultati.
    
    Args:
        session: requests.Session instance
        query: str query di ricerca
    
    Returns:
        str risposta sintetizzata
    """
    # API_KEY = os.getenv("OLLAMA_WEB_SEARCH_KEY")

    # 1) ricerca web Ollama
    """     r = session.post(
            "https://ollama.com/api/web_search",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "query": query,
                "max_results": MAX_RESULTS
            },
            timeout=20
        ) """
    
    r = session.post(
        "http://localhost:8888/search",
        params={
            "q": query,
            "format": "json",
            "lang": cfg.get("search_lang", "en-US"),
             "categories": "general"
        },
        timeout=20
    )    
    
    r.raise_for_status()
    data = r.json()

    # 2) prepara contesto
    context = "\n\n".join(
        f"{x['title']}\n{x['content'][:1500]}"
        for x in data.get("results", [])[:MAX_RESULTS]
    )

    # 3) Qwen3 sintetizza
    response = ollama.chat(
        model=LIBRARY,
        messages=[
            {
                "role": "system",
                "content": cfg.get("search_agent", "you are an assistant who uses up-to-date web sources.")
            },
            {
                "role": "user",
                "content": f"""
    {cfg.get("prompt_sources_label", "Answer the question using these sources:")}

    {context}

    {cfg.get("prompt_query_label", "Question:")}
    {query}
    """
            }
        ]
    )

    return response["message"]["content"]
