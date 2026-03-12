# Local Voice Assistant — Project Notes

## What This Is

A fully local, offline voice assistant running on a Windows workstation.
No cloud. No API keys. Everything runs on-device.

**Pipeline:** Wake Word → STT → LLM → TTS

```
Mic → openWakeWord ("Hey Jarvis")
    → faster-whisper large-v3  (speech-to-text, CUDA)
    → Ollama / llama3.1:70b    (language model, CUDA)
    → Piper en_US-lessac       (text-to-speech)
    → Speakers
```

## Hardware

- **GPU:** NVIDIA RTX A6000 (48 GB VRAM)
- **Mic:** AT2020USB+
- **OS:** Windows 11

## How to Run

Make sure Ollama is running (it auto-starts on Windows after install), then:

```bat
cd B:\Github\JohnTwenty\Computer
python main.py
```

Say **"Hey Jarvis"** to activate. Say "goodbye" to exit. Ctrl+C also works.

## Key Config (config.yaml)

- `wake_word.threshold: 0.95` — Had to go high to avoid false triggers from keyboard noise. If it stops responding to voice, lower toward 0.85.
- `llm.model: computer-assistant` — Custom Ollama model based on llama3.1:70b with `num_ctx 8192`. **Critical:** the default llama3.1:70b uses a 131k context window whose KV cache overflows the A6000, causing 59% CPU offload. The 8192-token context keeps everything 100% on GPU at ~27 tokens/sec.
- `stt.model: large-v3` — Best Whisper quality. Fits easily alongside the LLM on the A6000.

## Lessons Learned / Gotchas

### Ollama context window is the #1 performance trap
`ollama ps` showed `59%/41% CPU/GPU` on the stock llama3.1:70b. Root cause: the default 131,072-token context window creates a massive KV cache that exhausts VRAM before all model layers load. Fix: create a custom Ollama model with `num_ctx 8192`:
```
FROM llama3.1:70b
PARAMETER num_ctx 8192
```
```bat
ollama create computer-assistant -f Modelfile
```
Result: `100% GPU`, ~3-4x faster responses.

### openWakeWord requires ONNX models, not tflite
The package ships without bundled models. Call `download_models()` first, then load the `.onnx` file by full path — not by name — to avoid a hard dependency on `tflite-runtime` (which doesn't install cleanly on Python 3.12 / Windows):
```python
from openwakeword.utils import download_models
download_models(['hey_jarvis_v0.1'])
# Then load: Model(wakeword_models=['path/to/hey_jarvis_v0.1.onnx'])
```

### piper-tts 1.4.x has a completely new API
The old wave-file based `synthesize(text, wav_writer)` API is gone. Version 1.4+ returns `AudioChunk` objects with a `audio_float_array` (float32, already normalized):
```python
chunks = [chunk.audio_float_array for chunk in voice.synthesize(text)]
audio = np.concatenate(chunks)
```

### Ollama path on Windows
Not on PATH after winget install. Use:
```python
%LOCALAPPDATA%\Programs\Ollama\ollama.exe
```
Or add it to PATH manually.

## Models in Use

| Component | Model | Location |
|-----------|-------|----------|
| Wake word | hey_jarvis_v0.1.onnx | openwakeword package resources dir |
| STT | Whisper large-v3 | Auto-downloaded by faster-whisper to HuggingFace cache |
| LLM | llama3.1:70b (via computer-assistant) | Ollama model store |
| TTS | en_US-lessac-medium.onnx | models/ |

## Installed Packages

See `requirements.txt`. Key versions that worked:
- `faster-whisper 1.2.1`
- `openwakeword 0.6.0`
- `piper-tts 1.4.1`
- `onnxruntime 1.24.3`
- `sounddevice 0.5.5`
- Python 3.12.6, PyTorch 2.5.1+cu118

## Obvious Next Steps

- **Custom wake word "Computer"** (Star Trek style): train via Docker using [CoreWorxLab/openwakeword-training](https://github.com/CoreWorxLab/openwakeword-training). Needs a small update to `wake_word.py` to look in `models/` for custom `.onnx` files in addition to the package resources dir.
- **Tool use / agentic actions**: add a tool-calling layer to the LLM so it can search, run code, read files, etc. OpenClaw integration would be a natural fit here.
- **Streaming TTS latency**: currently synthesizes each sentence blocking. Could overlap synthesis of sentence N+1 while sentence N plays using a background thread + audio queue.
- **Linux portability**: all code is cross-platform Python. On Linux, `sounddevice` uses ALSA/PulseAudio automatically. Ollama path would just be `ollama` (on PATH after `install.sh`).
- **Voice switching**: several other Piper voices available via `python setup_models.py --list-voices`. British male (`en_GB-alan-medium`) sounds quite good.
