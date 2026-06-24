import json
import time
import requests

# Configurazione del tuo ambiente Ollama
URL = "http://127.0.0.1:11434/api/chat"
LIBRARY = "llama3.1:8b"  # Cambia con la stringa esatta della tua variabile LIBRARY

# 1. Simulazione dei tuoi file JSON di configurazione (Italiano)
prompt_templates = {
    "VECCHIO_PROMPT": {
        "identity_llm": "Nota importante: io sono Emma, l'assistente vocale personale di Marco. ",
        "identity_user": "Tu sei Marco, l'umano che interagisce con Emma.",
        "formatting_rules": "",
        "behavior_rules": ""
    },
    "NUOVO_PROMPT": {
        "identity_llm": "Sei Emma, l'assistente vocale personale di Marco.",
        "identity_user": "L'utente con cui parli è Marco.",
        "formatting_rules": "Rispondi sempre in modo naturale per una conversazione vocale. Usa un unico paragrafo senza elenchi o formattazioni. Preferisci frasi brevi e chiare. Sii concisa, ma non eliminare informazioni importanti. Evita risposte troppo lunghe.",
        "behavior_rules": "Non inventare mai informazioni personali, appuntamenti, ricordi o eventi passati. Se una richiesta riguarda dati che non possiedi, chiedi chiarimenti oppure dichiara che non hai queste informazioni."
    }
}

# 2. Domanda di test critica (forziamo il modello a dover gestire elenchi e numeri)
USER_INPUT = "Emma, Vamos a bailar esta vida nueva"

def invia_a_llama(system_prompt, user_text):
    # Costruiamo la history finta per il test con il system prompt aggiornato
    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text}
    ]
    
    payload = {
        "model": LIBRARY,
        "messages": history,
        "stream": False,
        "keep_alive": -1,
        "options": {
            "temperature": 0.3
        }
    }
    
    start_time = time.time()
    try:
        response = requests.post(URL, json=payload)
        response.raise_for_status()
        data = response.json()
        
        output = data["message"]["content"]
        durata = time.time() - start_time
        return output, durata
    except Exception as e:
        return f"Errore nella chiamata a Ollama: {e}", 0

# 3. Ciclo di confronto stampato su console
print("=" * 80)
print(f"TEST CONFRONTO PROMPT - OLLAMA API")
print(f"INPUT UTENTE: '{USER_INPUT}'")
print("=" * 80 + "\n")

for nome_prompt, chiavi in prompt_templates.items():
    # Unione dinamica delle chiavi come da tuo modello JSON
    
   
    system_prompt_completo = f"""
    [IDENTITÀ ASSISTENTE]
    {chiavi["identity_llm"], 'Sei Emma-Zira, un assistente vocale personale.'}

    [IDENTITÀ UTENTE]
    {chiavi["identity_user"], 'L utente è Marco.'}

    [COMPORTAMENTO]
    {chiavi["formatting_rules"], ''}

    [STILE RISPOSTA]
    {chiavi["behavior_rules"], ''}
    """    
    
    print (f"Prompt costruito: {system_prompt_completo}")

    print(f"🚀 ESECUZIONE CON: {nome_prompt}")
    print("-" * 40)
    
    risposta, tempo = invia_a_llama(system_prompt_completo, USER_INPUT)
    
    print(risposta)
    print("-" * 40)
    print(f"⏱️ Tempo totale di elaborazione: {tempo:.2f} secondi\n")

print("=" * 80)
