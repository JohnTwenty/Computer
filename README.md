# Computer — Local Voice Assistant

A fully offline, private voice assistant for your workstation.
No cloud. No API keys. Everything runs on your own hardware.

**Pipeline:**
```
Mic → Wake Word ("Computer") → Whisper STT → Ollama LLM → Piper TTS → Speakers
```

---

## Requirements

- Python 3.10+
- NVIDIA GPU with 8+ GB VRAM (48 GB recommended for the 70B model)
- CUDA drivers installed
- A microphone and speakers

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/JohnTwenty/Computer.git
cd Computer
pip install -r requirements.txt
```

### 2. Install Ollama

**Windows:**
```bat
winget install Ollama.Ollama
```

**Linux/macOS:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 3. Pull a language model

```bash
# 70B — best quality, requires ~48 GB VRAM
ollama pull llama3.1:70b

# 8B — good quality, runs on 8 GB VRAM
ollama pull llama3.1:8b
```

Then create the assistant model with a reduced context window (keeps it 100% on GPU):

```bash
ollama create computer-assistant -f Modelfile
```

### 4. Download voice and wake word models

```bash
python setup_models.py
```

This downloads:
- Piper TTS voice model → `models/`
- openWakeWord "Computer" wake word model → `models/`

### 5. Run

Make sure Ollama is running (`ollama serve` on Linux/macOS, auto-starts on Windows), then:

```bash
python main.py
```

Say **"Computer"** to activate. Say **"goodbye"** to exit. Say **"cancel"** to dismiss a false trigger.

---

## Configuration

All settings are in `config.yaml`:

| Setting | Default | Description |
|---|---|---|
| `wake_word.model` | `computer_v2` | Wake word model (filename in `models/`) |
| `wake_word.threshold` | `0.98` | Detection confidence (0–1). Higher = less sensitive |
| `wake_word.trigger_level` | `3` | Consecutive frames required to activate. Debounces noise spikes |
| `stt.model` | `large-v3` | Whisper model size (`tiny` / `base` / `small` / `medium` / `large-v3`) |
| `llm.model` | `computer-assistant` | Ollama model name |
| `audio.silence_threshold` | `0.02` | RMS level below which audio is considered silence |
| `audio.silence_duration` | `1.5` | Seconds of silence before recording stops |

### Tuning wake word sensitivity

If you get false positives from background noise:
1. Raise `threshold` (max ~0.99)
2. Raise `trigger_level` (3–5 works well for most environments)

If it stops responding to your voice:
1. Lower `threshold` toward 0.85

---

## Components

| Layer | Library | Notes |
|---|---|---|
| Wake word | [openWakeWord](https://github.com/dscripka/openWakeWord) | ONNX, runs on CPU |
| Wake word model | [fwartner/home-assistant-wakewords-collection](https://github.com/fwartner/home-assistant-wakewords-collection) | "Computer" v2 |
| Speech-to-text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | GPU accelerated |
| LLM | [Ollama](https://ollama.com) + llama3.1:70b | 100% local |
| Text-to-speech | [Piper](https://github.com/rhasspy/piper) | `en_US-lessac-medium` |
| Audio I/O | [sounddevice](https://python-sounddevice.readthedocs.io) | Cross-platform |

---

## Changing the wake word

Other pre-trained options from openWakeWord:

```bash
python -c "from openwakeword.utils import download_models; download_models(['hey_jarvis_v0.1'])"
```

Then set `wake_word.model: hey_jarvis_v0.1` in `config.yaml`.

Available built-ins: `alexa_v0.1`, `hey_jarvis_v0.1`, `hey_mycroft_v0.1`, `hey_rhasspy_v0.1`

---

## Running on a smaller GPU

Edit `config.yaml`:
```yaml
stt:
  model: "small"          # instead of large-v3
  compute_type: "int8"    # instead of float16

llm:
  model: "llama3.1:8b"    # pull with: ollama pull llama3.1:8b
```

And update `Modelfile` to `FROM llama3.1:8b`.
