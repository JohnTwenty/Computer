"""
Downloads required models into the models/ directory.
Run once before first use: python setup_models.py

  - Piper TTS voice model  → models/
  - openWakeWord ONNX models → openwakeword package resources dir

Whisper models are downloaded automatically by faster-whisper on first run.
Ollama models are pulled via: ollama pull <model-name>
"""

import os
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path("models")

# Piper voices are hosted on Hugging Face.
# Format: (filename, hf_path)
PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

PIPER_VOICES = {
    "en_US-lessac-medium": {
        "hf_dir": "en/en_US/lessac/medium",
        "files": [
            "en_US-lessac-medium.onnx",
            "en_US-lessac-medium.onnx.json",
        ],
    },
    "en_US-amy-medium": {
        "hf_dir": "en/en_US/amy/medium",
        "files": [
            "en_US-amy-medium.onnx",
            "en_US-amy-medium.onnx.json",
        ],
    },
    "en_GB-alan-medium": {
        "hf_dir": "en/en_GB/alan/medium",
        "files": [
            "en_GB-alan-medium.onnx",
            "en_GB-alan-medium.onnx.json",
        ],
    },
}

DEFAULT_VOICE = "en_US-lessac-medium"


def download_file(url: str, dest: Path):
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return
    print(f"  Downloading {dest.name} ...", end="", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
        print(f" done ({dest.stat().st_size // 1024} KB)")
    except Exception as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)


def download_voice(voice_name: str):
    if voice_name not in PIPER_VOICES:
        print(f"Unknown voice '{voice_name}'. Available: {list(PIPER_VOICES)}")
        sys.exit(1)

    info = PIPER_VOICES[voice_name]
    print(f"\nDownloading Piper voice: {voice_name}")
    for filename in info["files"]:
        url = f"{PIPER_BASE}/{info['hf_dir']}/{filename}"
        dest = MODELS_DIR / filename
        download_file(url, dest)


COMPUTER_WAKE_WORD_URL = (
    "https://raw.githubusercontent.com/fwartner/home-assistant-wakewords-collection"
    "/main/en/computer/computer_v2.onnx"
)


def download_wake_word_models(model_name: str = "hey_jarvis_v0.1"):
    """Download openWakeWord ONNX models (feature models + the named wake word).
    For the 'computer_v2' model, fetches directly from the community collection repo.
    """
    # Always download the openWakeWord feature/embedding models (needed by all wake words)
    print(f"\nDownloading openWakeWord feature models...")
    from openwakeword.utils import download_models
    download_models([])  # empty list = feature models only
    print("  Feature models OK.")

    if model_name == "computer_v2":
        print(f"\nDownloading wake word model: computer_v2 (community model)")
        download_file(COMPUTER_WAKE_WORD_URL, MODELS_DIR / "computer_v2.onnx")
    else:
        print(f"\nDownloading openWakeWord model: {model_name}")
        download_models([model_name])


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download all required models")
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        choices=list(PIPER_VOICES),
        help=f"Piper voice to download (default: {DEFAULT_VOICE})",
    )
    parser.add_argument(
        "--wake-word",
        default="computer_v2",
        help="openWakeWord model to download (default: computer_v2)",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available Piper voices and exit",
    )
    args = parser.parse_args()

    if args.list_voices:
        print("Available Piper voices:")
        for name in PIPER_VOICES:
            print(f"  {name}")
        return

    download_voice(args.voice)
    download_wake_word_models(args.wake_word)

    print("\nAll models downloaded.")
    if args.voice != DEFAULT_VOICE:
        print(f"Update config.yaml: tts.model_path: models/{args.voice}.onnx")
    if args.wake_word != "computer_v2":
        print(f"Update config.yaml: wake_word.model: {args.wake_word}")


if __name__ == "__main__":
    main()
