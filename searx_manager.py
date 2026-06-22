import subprocess
import time
import requests


# Variabile globale per conservare il processo WSL avviato
_searx_process = None


def is_searxng_running():
    """
    Verifica se SearXNG risponde sulla porta 8888.
    """
    try:
        response = requests.get(
            "http://localhost:8888",
            timeout=2
        )
        return response.status_code < 500

    except Exception:
        return False


def start_searxng():
    """
    Avvia SearXNG dentro WSL in background.
    Se è già attivo non fa nulla.
    """

    global _searx_process

    if is_searxng_running():
        print("[SEARXNG] Già attivo.")
        return

    comando_linux = (
        "cd ~/searxng && "
        "source .venv/bin/activate && "
        "python searx/webapp.py --port 8888 --bind 0.0.0.0"
    )

    comando_wsl = [
        "wsl.exe",
        "-d",
        "Ubuntu",
        "--exec",
        "bash",
        "-c",
        comando_linux
    ]

    try:
        print("[SEARXNG] Avvio in corso...")

        _searx_process = subprocess.Popen(
            comando_wsl,
            # commentarli per vedere l'output del server SearXNG
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        # piccolo tempo per permettere al server di inizializzarsi
        for _ in range(20):
            if is_searxng_running():
                print("[SEARXNG] Avviato.")
                return

            time.sleep(0.5)

        print("[SEARXNG] Avvio richiesto, ma non ancora raggiungibile.")

    except Exception as e:
        print(f"[SEARXNG] Errore avvio: {e}")


def stop_searxng():
    """
    Arresta SearXNG.
    """

    global _searx_process

    try:
        print("[SEARXNG] Arresto in corso...")

        comando_linux = "pkill -f 'python searx/webapp.py'"

        comando_wsl = [
            "wsl.exe",
            "-d",
            "Ubuntu",
            "--exec",
            "bash",
            "-c",
            comando_linux
        ]

        subprocess.run(
            comando_wsl,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        _searx_process = None

        print("[SEARXNG] Arrestato.")

    except Exception as e:
        print(f"[SEARXNG] Errore arresto: {e}")