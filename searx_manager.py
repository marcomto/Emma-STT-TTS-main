import subprocess
import time
import requests


# Global variable to store the started WSL process
_searx_process = None


def is_searxng_running():
    """
    Checks if SearXNG is responding on port 8888.
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
    Starts SearXNG inside WSL in the background.
    If it is already running, it does nothing.
    """

    global _searx_process

    if is_searxng_running():
        print("[SEARXNG] Already running.")
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
        print("[SEARXNG] Starting...")

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
                print("[SEARXNG] Started.")
                return

            time.sleep(0.5)

        print("[SEARXNG] Start requested, but not yet reachable.")

    except Exception as e:
        print(f"[SEARXNG] Start error: {e}")


def stop_searxng():
    """
    Stops SearXNG.
    """

    global _searx_process

    try:
        print("[SEARXNG] Stop in progress...")

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

        print("[SEARXNG] Stopped.")

    except Exception as e:
        print(f"[SEARXNG] Stop error: {e}")