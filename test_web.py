
# Se usi Ollama Cloud/API compatibile:
#OLLAMA_API_KEY = "c319f8900a6d482993f8a1c375d5ca72.jqzeg1JHJFzskBhGHC0-RWj6"
import requests
import ollama

API_KEY = "c319f8900a6d482993f8a1c375d5ca72.jqzeg1JHJFzskBhGHC0-RWj6"

query = "quali sono le attrattive per il comune di san mauro torinese, torino, italia.?"

# 1) ricerca web Ollama
r = requests.post(
    "https://ollama.com/api/web_search",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "query": query,
        "max_results": 5
    }
)

data = r.json()

# 2) prepara contesto
context = "\n\n".join(
    f"{x['title']}\n{x['url']}\n{x.get('content','')}"
    for x in data["results"]
)

# 3) Qwen3 sintetizza
response = ollama.chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "system",
            "content": "Sei un assistente che usa fonti web aggiornate."
        },
        {
            "role": "user",
            "content": f"""
Rispondi alla domanda usando queste fonti:

{context}

Domanda:
{query}
"""
        }
    ]
)

print(response["message"]["content"])