"""
Audio I/O manager.
Uses sounddevice for cross-platform mic input and speaker output.
"""

import queue
import numpy as np
import sounddevice as sd


class AudioManager:
    def __init__(self, config):
        cfg = config["audio"]
        self.sample_rate = cfg.get("sample_rate", 16000)
        self.channels = cfg.get("channels", 1)
        self.chunk_size = cfg.get("chunk_size", 1280)
        self.silence_threshold = cfg.get("silence_threshold", 0.02)
        self.silence_duration = cfg.get("silence_duration", 1.5)
        self.max_record_duration = cfg.get("max_record_duration", 30.0)

        self._queue: queue.Queue = queue.Queue()
        self._stream = None

    def start(self):
        """Open the microphone input stream."""
        def _callback(indata, frames, time_info, status):
            self._queue.put(indata.copy().flatten())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.chunk_size,
            callback=_callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read_chunk(self) -> np.ndarray:
        """Block until the next audio chunk is available; return int16 array."""
        return self._queue.get()

    def drain(self):
        """Discard any audio buffered while the wake word was being processed."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def record_until_silence(self) -> np.ndarray:
        """
        Read from the live stream until silence is detected.
        Waits for initial speech before starting the silence timer.
        Returns float32 audio normalized to [-1.0, 1.0].
        """
        chunks = []
        silent_chunks = 0
        chunks_per_second = self.sample_rate / self.chunk_size
        required_silent = int(self.silence_duration * chunks_per_second)
        max_chunks = int(self.max_record_duration * chunks_per_second)
        speech_detected = False

        for _ in range(max_chunks):
            chunk = self.read_chunk()
            chunks.append(chunk)

            rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2)) / 32768.0

            if rms > self.silence_threshold:
                speech_detected = True
                silent_chunks = 0
            elif speech_detected:
                silent_chunks += 1

            if speech_detected and silent_chunks >= required_silent:
                break

        audio = np.concatenate(chunks).astype(np.float32) / 32768.0
        return audio

    def play(self, audio: np.ndarray, sample_rate: int = None):
        """Play a float32 numpy audio array through the default output device."""
        sd.play(audio, sample_rate or self.sample_rate, blocking=True)

    def play_acknowledgment(self):
        """Play a short double-beep to signal wake word detected."""
        sr = self.sample_rate
        fade = int(0.015 * sr)

        def _beep(freq, duration):
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            tone = np.sin(2 * np.pi * freq * t).astype(np.float32) * 0.35
            tone[:fade] *= np.linspace(0, 1, fade)
            tone[-fade:] *= np.linspace(1, 0, fade)
            return tone

        gap = np.zeros(int(0.04 * sr), dtype=np.float32)
        sound = np.concatenate([_beep(880, 0.09), gap, _beep(1100, 0.09)])
        sd.play(sound, sr, blocking=True)
