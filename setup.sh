#!/usr/bin/env bash
set -e

echo "============================================================"
echo " Local Voice Assistant - Linux/macOS Setup"
echo "============================================================"
echo

echo "[1/3] Installing Python dependencies..."
pip install -r requirements.txt

echo
echo "[2/3] Downloading Piper TTS voice model..."
python setup_models.py

echo
echo "[3/3] Checking Ollama..."
if ! command -v ollama &>/dev/null; then
    echo
    echo "  Ollama is NOT installed."
    echo "  Install with:  curl -fsSL https://ollama.com/install.sh | sh"
    echo "  Then run:      ollama pull llama3.1:70b"
    echo
else
    echo "  Ollama found."
    echo "  If you haven't already:  ollama pull llama3.1:70b"
fi

echo
echo "============================================================"
echo " Setup complete."
echo " Start Ollama:    ollama serve"
echo " Run assistant:   python main.py"
echo "============================================================"
