import threading
import pyaudio
from piper import PiperVoice

class TTS:
    # Passa l'istanza 'pa' al costruttore
    def __init__(self, model_path, pa_instance):
        self.voice = PiperVoice.load(model_path)
        print("Piper sample rate:", self.voice.config.sample_rate)
        self.stop_event = threading.Event()
        
        # Usa l'istanza condivisa passata come argomento
        self.pa = pa_instance 
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.voice.config.sample_rate,
            output=True
        )
        self.thread = None

    def _play_loop(self, text):
        """Metodo interno eseguito nel thread secondario."""
        # Svuota i buffer residui prima di iniziare
        try:
            self.stream.stop_stream()
            self.stream.start_stream()
        except Exception:
            pass

        for audio_chunk in self.voice.synthesize(text):
            if self.stop_event.is_set():
                break
            try:
                self.stream.write(audio_chunk.audio_int16_bytes)
            except Exception:
                break

    def start(self, text):
        """Avvia la riproduzione in un thread separato."""
        self.stop() # Ferma eventuale audio precedente
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._play_loop, args=(text,), daemon=True)
        self.thread.start()

    def stop(self):
        """Interrompe immediatamente l'audio."""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join() # Attende la chiusura pulita del thread

    def is_speaking(self):
        """Verifica se il thread sta ancora riproducendo."""
        return self.thread is not None and self.thread.is_alive()

    def close(self):
        try:
            self.stop()
            self.stream.stop_stream()
            self.stream.close()
            # NOTA: Non chiamare self.pa.terminate() qui se è condivisa, 
            # altrimenti chiuderai l'audio anche per il microfono.
        except Exception:
            pass