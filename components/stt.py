"""
Speech-to-text using faster-whisper (CTranslate2 backend, CUDA-accelerated).
"""

import numpy as np
from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, config):
        cfg = config["stt"]
        model_size = cfg.get("model", "large-v3")
        device = cfg.get("device", "cuda")
        compute_type = cfg.get("compute_type", "float16")

        print(f"Loading Whisper '{model_size}' on {device} ({compute_type})...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = cfg.get("language", "en")
        print("Whisper ready.")

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe float32 audio (16 kHz, mono, normalized to [-1, 1]).
        Returns stripped transcript string, or empty string if nothing detected.
        """
        segments, _info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
