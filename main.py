"""
Local Voice Assistant
Pipeline: wake word → STT (Whisper) → LLM (Ollama) → TTS (Piper)

Usage:
    python main.py [--config path/to/config.yaml]
"""

import argparse
import sys

import yaml

from components.audio import AudioManager
from components.wake_word import WakeWordDetector
from components.stt import Transcriber
from components.llm import LLMClient
from components.tts import TTSSynthesizer

# Sentence-ending punctuation that triggers speaking the buffered sentence
SENTENCE_ENDS = {".", "?", "!", "\n"}

# Phrases that cause the assistant to exit gracefully
EXIT_PHRASES = {"quit", "exit", "goodbye", "bye", "shut down", "stop listening"}

# Phrases that cancel the current interaction and return to wake-word listening
CANCEL_PHRASES = {"cancel", "never mind", "nevermind", "abort", "forget it", "stop", "ignore"}

# Max conversation turns kept in context (each turn = 1 user + 1 assistant message)
MAX_HISTORY_TURNS = 10


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class VoiceAssistant:
    def __init__(self, config: dict):
        self.config = config
        self.history: list[dict] = []

        self.audio = AudioManager(config)
        self.wake_word = WakeWordDetector(config)
        self.stt = Transcriber(config)
        self.llm = LLMClient(config)
        self.tts = TTSSynthesizer(config)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _trim_history(self):
        """Keep only the most recent MAX_HISTORY_TURNS turns."""
        max_msgs = MAX_HISTORY_TURNS * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    def _stream_and_speak(self, messages: list[dict]) -> str:
        """
        Stream tokens from the LLM, speaking each sentence as it completes.
        Returns the full response text.
        """
        sentence_buf = ""
        full_response = ""

        print("Assistant: ", end="", flush=True)

        for token in self.llm.stream_response(messages):
            full_response += token
            sentence_buf += token
            print(token, end="", flush=True)

            # Speak when we hit a sentence boundary
            stripped = sentence_buf.rstrip()
            if stripped and stripped[-1] in SENTENCE_ENDS:
                self.tts.speak(stripped)
                sentence_buf = ""

        # Speak any trailing text that didn't end with punctuation
        if sentence_buf.strip():
            self.tts.speak(sentence_buf.strip())

        print()  # newline after streamed response
        return full_response

    # ------------------------------------------------------------------
    # Interaction loop
    # ------------------------------------------------------------------

    def handle_interaction(self) -> bool:
        """
        Handle one full interaction cycle: record → transcribe → respond.
        Returns False if the user asked to exit, True otherwise.
        """
        self.audio.play_acknowledgment()
        self.audio.drain()  # discard audio buffered during wake-word processing

        print("Listening...", flush=True)
        audio = self.audio.record_until_silence()

        text = self.stt.transcribe(audio)
        if not text:
            print("[No speech detected]")
            return True

        print(f"\nYou: {text}")

        if any(phrase in text.lower() for phrase in CANCEL_PHRASES):
            print("[Cancelled]")
            return True

        if any(phrase in text.lower() for phrase in EXIT_PHRASES):
            self.tts.speak("Goodbye!")
            return False

        self.history.append({"role": "user", "content": text})
        self._trim_history()

        response = self._stream_and_speak(self.history)
        self.history.append({"role": "assistant", "content": response})

        return True

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self):
        # Pre-flight: verify Ollama is reachable before loading audio
        print("\nChecking Ollama connection...")
        if not self.llm.check_connection():
            print(
                f"\nERROR: Cannot reach Ollama at {self.config['llm']['base_url']}\n"
                "  Start it with:  ollama serve\n"
                "  Then re-run this script."
            )
            sys.exit(1)
        print("Ollama OK.")

        self.audio.start()

        wake_label = self.wake_word.label
        print(f"\nReady — say \"{wake_label}\" to activate.  Ctrl+C to quit.\n")

        try:
            while True:
                chunk = self.audio.read_chunk()
                if self.wake_word.process(chunk):
                    print(f"\n[Wake word detected]")
                    if not self.handle_interaction():
                        break
                    print(f'\nListening for "{wake_label}"...')
        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            self.audio.stop()
            print("Shut down.")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Local voice assistant")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    assistant = VoiceAssistant(config)
    assistant.run()


if __name__ == "__main__":
    main()
