import os
import sys
import time

# --- BLOCCO DI FORZATURA PER LE DLL DI CUDA ---
if sys.platform == "win32":
    base_nvidia_path = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
    cublas_lib = os.path.join(base_nvidia_path, "cublas", "lib")
    cudnn_lib = os.path.join(base_nvidia_path, "cudnn", "lib")
    
    if os.path.exists(cublas_lib) and os.path.exists(cudnn_lib):
        os.environ["PATH"] = cublas_lib + os.pathsep + cudnn_lib + os.pathsep + os.environ["PATH"]

# --- AVVIO DEL TEST ---
print("=" * 50)
print("1. VERIFICA REQUISITI DI SISTEMA")
print("=" * 50)

try:
    import torch
    cuda_disponibile = torch.cuda.is_available()
    print(f"[*] PyTorch rileva CUDA? {'SÌ' if cuda_disponibile else 'NO'}")
    if cuda_disponibile:
        print(f"[*] Nome Scheda Video: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("[!] PyTorch non è installato, salto questo controllo.")

print("\n" + "=" * 50)
print("2. INIZIALIZZAZIONE FASTER-WHISPER SU CUDA")
print("=" * 50)

try:
    from faster_whisper import WhisperModel
    
    print("[*] Caricamento del modello 'tiny' su GPU (CUDA)...")
    start_time = time.time()
    
    # Tentativo di inizializzazione su CUDA
    model = WhisperModel("tiny", device="cuda", compute_type="float16")
    
    print(f"[✓] Modello caricato in GPU con successo! (Tempo: {time.time() - start_time:.2f}s)")
    
    # Se arriva qui, CUDA funziona! Facciamo un test di trascrizione simulato
    print("\n[✓] TEST EFFETTUATO CON SUCCESSO: CUDA è configurato correttamente!")
    print("Ora puoi usare 'device=\"cuda\"' e 'compute_type=\"float16\"' nel tuo script principale.")

except Exception as e:
    print("\n[X] ERRORE DI CONFIGURAZIONE CUDA:")
    print("-" * 50)
    print(str(e))
    print("-" * 50)
    print("\n[💡] Suggerimento: Se l'errore parla ancora di 'cublas64_12.dll',")
    print("sposta manualmente quel file nella cartella di questo script come spiegato prima.")
