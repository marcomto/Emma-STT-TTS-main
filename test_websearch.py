import requests
import argparse

session = requests.Session()
query = "caterina caselli"

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

    for r in data["results"][:10]:
        print(r.get("engine"), "->", r.get("title"))  

    #print(data)

    # 2) prepara contesto
    context = "\n\n".join(
        f"{x.get('title','')}\n{x.get('content','')[:1500]}"
        for x in data.get("results", [])[:max_results]
    )
    print(context)
    return context


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test searXNG web search snippet")
    parser.add_argument("--max-results", type=int, default=5, help="Numero massimo di risultati da includere nel contesto")
    parser.add_argument("--query", type=str, default=None, help="Query di ricerca (sovrascrive la query hardcoded)")
    args = parser.parse_args()

    q = args.query if args.query else query
    result = web_search(session, q, max_results=args.max_results)
    if not result:
        print("Nessun risultato trovato o risposta vuota.")



