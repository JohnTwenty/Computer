"""
Local Voice Assistant
Pipeline: wake word → STT (Whisper) → LLM (Ollama) → TTS (Piper)

Usage:
    python main.py [--config path/to/config.yaml]
"""

import argparse
import sys

import requests
import yaml

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
        from components.audio import AudioManager
        from components.wake_word import WakeWordDetector
        from components.stt import Transcriber
        from components.llm import LLMClient
        from components.tts import TTSSynthesizer

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

def _parse_num_ctx(parameters: str) -> str:
    """Extract num_ctx value from an Ollama parameters string, or return 'default'."""
    for line in parameters.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "num_ctx":
            return parts[1]
    return "default"


def fetch_ollama_models(base_url: str) -> list[dict]:
    """Return sorted list of model info dicts with name, param_size, and ctx."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot reach Ollama at {base_url}\n  Start it with: ollama serve")
        sys.exit(1)

    models = []
    for m in sorted(r.json().get("models", []), key=lambda x: x["name"]):
        name = m["name"]
        param_size = m.get("details", {}).get("parameter_size", "?")

        show = requests.post(f"{base_url}/api/show", json={"name": name}, timeout=5)
        ctx = _parse_num_ctx(show.json().get("parameters", "")) if show.ok else "?"

        models.append({"name": name, "param_size": param_size, "ctx": ctx})
    return models


def _format_model_line(m: dict) -> str:
    return f"{m['name']}  {m['param_size']}  ctx:{m['ctx']}"


def choose_model_interactively(base_url: str, current: str) -> str:
    """Print a numbered list of Ollama models and prompt the user to pick one."""
    models = fetch_ollama_models(base_url)
    if not models:
        print("No models found in Ollama. Pull one first, e.g.: ollama pull qwen2.5:7b")
        sys.exit(1)

    print("\nAvailable models:")
    for i, m in enumerate(models, 1):
        marker = "  <-- current" if m["name"] == current else ""
        print(f"  {i}.  {_format_model_line(m)}{marker}")

    while True:
        raw = input(f"\nSelect model [1-{len(models)}], or press Enter to keep '{current}': ").strip()
        if raw == "":
            return current
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            return models[int(raw) - 1]["name"]
        print(f"  Please enter a number between 1 and {len(models)}.")


def main():
    parser = argparse.ArgumentParser(description="Local voice assistant")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--model", metavar="NAME", help="Override the Ollama model from config")
    parser.add_argument("--list-models", action="store_true", help="List available Ollama models and exit")
    parser.add_argument("--choose", action="store_true", help="Interactively choose a model before starting")
    args = parser.parse_args()

    config = load_config(args.config)
    base_url = config["llm"].get("base_url", "http://localhost:11434").rstrip("/")

    if args.list_models:
        current = config["llm"].get("model", "")
        print("\nAvailable Ollama models:")
        for m in fetch_ollama_models(base_url):
            marker = "  <-- current" if m["name"] == current else ""
            print(f"  {_format_model_line(m)}{marker}")
        return

    if args.model:
        config["llm"]["model"] = args.model
    elif args.choose:
        config["llm"]["model"] = choose_model_interactively(base_url, config["llm"].get("model", ""))

    assistant = VoiceAssistant(config)
    assistant.run()


if __name__ == "__main__":
    main()
