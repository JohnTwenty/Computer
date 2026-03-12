"""
Text-to-speech using Piper (piper-tts Python package, ONNX backend).
piper-tts >= 1.4.0: synthesize() yields AudioChunk objects with float32 audio arrays.
"""

import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice


class TTSSynthesizer:
    def __init__(self, config):
        cfg = config["tts"]
        model_path = cfg.get("model_path", "models/en_US-lessac-medium.onnx")

        print(f"Loading Piper TTS model: {model_path}")
        self.voice = PiperVoice.load(model_path)
        self.sample_rate = self.voice.config.sample_rate
        self.speaker_id = cfg.get("speaker_id", None)
        print(f"TTS ready (sample rate: {self.sample_rate} Hz)")

    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize text to a float32 numpy array at self.sample_rate.
        Each AudioChunk.audio_float_array is already float32 in [-1, 1].
        """
        chunks = [chunk.audio_float_array for chunk in self.voice.synthesize(text)]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    def speak(self, text: str):
        """Synthesize text and play it immediately (blocking)."""
        text = text.strip()
        if not text:
            return
        audio = self.synthesize(text)
        sd.play(audio, self.sample_rate, blocking=True)
