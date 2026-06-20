import time
import requests
import json
import os
import pyttsx3
import winsound

session = requests.Session()


def create_tts_wav(text: str, filename: str = "response.wav") -> str:
    """Generate a WAV file from text using pyttsx3 with an Italian voice."""
    engine = pyttsx3.init()

    italian_voice_id = None
    for voice in engine.getProperty("voices"):
        name = voice.name.lower()
        languages = "".join(getattr(voice, "languages", [])).lower()
        if "ital" in name or "ital" in languages or "it_" in languages or "italian" in languages:
            italian_voice_id = voice.id
            break

    if italian_voice_id:
        engine.setProperty("voice", italian_voice_id)
    else:
        print("[WARN] Voce italiana non trovata, verrà usata la voce di default.")

    engine.save_to_file(text, filename)
    engine.runAndWait()
    engine.stop()
    return filename


def play_wav(filename: str):
    """Play the WAV file on Windows using winsound."""
    if not os.path.exists(filename):
        print(f"[WARN] WAV file not found: {filename}")
        return

    winsound.PlaySound(filename, winsound.SND_FILENAME)


def call_ollama(history):

    url = "http://127.0.0.1:11434/api/chat"
    
   
    payload = {
        "model": "llama3.1:8b", # o il nome esatto del modello Llama che stai usando
        "messages": history,
        "stream": False,
        "keep_alive": -1,    # Lascialo attivo: ora sta funzionando perfettamente!
        "options": {
            "temperature": 0.3
        }
    }    

    start = time.time()
    r = session.post(url, json=payload, timeout=45)
    end = time.time()

    elapsed = end - start
    llm_resp_time = "LLM response"
    payload_chars = "chars"

    print(f"[DEBUG][call_ollama] {llm_resp_time}: {elapsed:.2f}s (payload {len(json.dumps(payload))} {payload_chars})")

    if r.status_code != 200:
        print(f"[WARN] Ollama HTTP {r.status_code}")
        return {"role": "assistant", "content": "non ho capito"}

    data = r.json()
    assistant_text = data.get("message", {}).get("content", "")
    print(assistant_text)

    if assistant_text:
        wav_file = create_tts_wav(assistant_text, "ollama_response.wav")
        play_wav(wav_file)

    return {"role": "assistant", "content": assistant_text}


history_corretta = [{'role': 'user', 'content': 'come si preparano gli spaghetti alla carbonara IN BREVE.'}]

call_ollama(history_corretta)



