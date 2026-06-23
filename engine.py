import threading
import pyaudio
from piper import PiperVoice

class TTS:
    # Pass the 'pa' instance to the constructor
    def __init__(self, model_path, pa_instance):
        self.voice = PiperVoice.load(model_path)
        print("Piper sample rate:", self.voice.config.sample_rate)
        self.stop_event = threading.Event()
        
        # Use the shared instance passed as an argument
        self.pa = pa_instance 
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.voice.config.sample_rate,
            output=True
        )
        self.thread = None
        self.finished_event = threading.Event()

    def _play_loop(self, text):

        self.finished_event.clear()

        try:

            BLOCK_SIZE = 2048

            for audio_chunk in self.voice.synthesize(text):

                if self.stop_event.is_set():
                    break

                data = audio_chunk.audio_int16_bytes

                for i in range(0, len(data), BLOCK_SIZE):

                    if self.stop_event.is_set():
                        break

                    self.stream.write(data[i:i + BLOCK_SIZE])

                if self.stop_event.is_set():
                    break

        finally:
            self.finished_event.set()


    def start(self, text):
        """Starts the playback in a separate thread."""
        self.stop() # Stop any previous audio
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._play_loop, args=(text,), daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the audio immediately."""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join() # Waits for the thread to close cleanly

    def is_speaking(self):
        """Verifies if the thread is still playing."""
        return self.thread is not None and self.thread.is_alive()

    def close(self):
        try:
            self.stop()
            self.stream.stop_stream()
            self.stream.close()
            # Don't call self.pa.terminate() here if it's shared,
            # otherwise you'll mute the microphone as well.
        except Exception:
            pass