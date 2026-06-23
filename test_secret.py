import os
from dotenv import load_dotenv

load_dotenv("secret.env")
# Ora la chiave è caricata in memoria, ma non è scritta nel codice!
api_key = os.getenv("OLLAMA_WEB_SEARCH_KEY")
