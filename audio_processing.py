"""
Audio processing utilities for transcription and voice recognition.
"""
import numpy as np
from faster_whisper import WhisperModel
from load_config import cfg


def transcribe_audio(frames, whisper_model):
    """
    Transcribe audio frames using Whisper model.
    
    Args:
        frames: List of audio frame bytes
        whisper_model: WhisperModel instance
        
    Returns:
        Transcribed text (lowercase, stripped)
    """
    audio_bytes = b"".join(frames)

    audio = np.frombuffer(
        audio_bytes,
        np.int16
    ).astype(np.float32)

    audio /= 32768.0

    segments, info = whisper_model.transcribe(
        audio,
        language=cfg.get("user_lang"),
        vad_filter=True
    )

    text = "".join(segment.text for segment in segments)

    return text.lower().strip()
