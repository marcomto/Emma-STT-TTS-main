import requests
import argparse
import ollama_client as ollama

session = requests.Session()
query = "la luce verde di Scott Fitzgerald"
def web_search(session, query, max_results=5):


    # 1) ricerca web Ollama
    r = session.post(
        "http://localhost:8888/search",
        params={
            "q": query,
            "format": "json",
            "lang": "it-IT",
             "categories": "general"
        },
        timeout=20
    )

 
    
    r.raise_for_status()
    data = r.json()

    #print(data)

    # 2) prepara contesto
    context = """Rispondi alla domanda dell'utente utilizzando solo le informazioni pertinenti ricavate dai risultati della ricerca web qui sopra. Mantieni la risposta concisa e chiara:"""
    context += "\n\n".join(
        f"{x.get('title','')}\n{x.get('content','')[:1500]}"
        for x in data.get("results", [])[:max_results]
    )
    context += "Domanda utente:" + query
    #print(context)

    url = "http://127.0.0.1:11434/api/chat"
    
    messages = []
    messages.append({"role": "system", "content": """
        Sei un assistente vocale personale locale. Rispondi alla domanda dell'utente utilizzando i risultati della ricerca web forniti. Utilizza solo informazioni pertinenti. Ignora annunci pubblicitari, contenuti SEO, duplicati e risultati non correlati. Se le informazioni sono insufficienti, dillo chiaramente. Fornisci una risposta concisa e adatta alla voce. Non menzionare il processo di ricerca o le fonti a meno che non venga richiesto.
        """})
   
    messages.append({"role": "user", "content": context})

    #print(messages)

    payload = {
        "model": "llama3.1:8b",
        "messages": messages,
        "stream": False,
        "keep_alive": -1,
        "options": {
            "temperature": 0.3
        }
    }


    try:
        r = session.post(url, json=payload, timeout=45)
        #print(f"[DEBUG] Ollama response status: {r.status_code}")
        #   print(f"[DEBUG] Ollama response: {r.text[:500]}")  # primi 500 caratteri
    except requests.exceptions.ConnectionError as e:
        print(f"[ERROR] Connessione a Ollama fallita: {e}")
        return None
    except requests.exceptions.Timeout as e:
        print(f"[ERROR] Timeout Ollama: {e}")
        return None

    if r.status_code != 200:
        print(f"[WARN] Ollama HTTP {r.status_code}: {r.text}")
        return None

    data = r.json()
    
    return data["message"]["content"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test searXNG web search snippet")
    parser.add_argument("--max-results", type=int, default=5, help="Numero massimo di risultati da includere nel contesto")
    parser.add_argument("--query", type=str, default=None, help="Query di ricerca (sovrascrive la query hardcoded)")
    args = parser.parse_args()

    q = args.query if args.query else query
    result = web_search(session, q, max_results=args.max_results)
    if result:
        print(f"\n[RESULT]\n{result}")
    else:
        print("Nessun risultato trovato o risposta vuota.")



