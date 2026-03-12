@echo off
echo ============================================================
echo  Local Voice Assistant - Windows Setup
echo ============================================================
echo.

echo [1/3] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed. See output above.
    exit /b 1
)

echo.
echo [2/3] Downloading Piper TTS voice model...
python setup_models.py
if errorlevel 1 (
    echo ERROR: Model download failed.
    exit /b 1
)

echo.
echo [3/3] Checking Ollama...
set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if not exist "%OLLAMA%" (
    echo.
    echo  Ollama is NOT installed.
    echo  Install it with:  winget install Ollama.Ollama
    echo  Then run:         ollama pull llama3.1:70b
    echo.
) else (
    echo  Ollama found at %OLLAMA%
    echo  If you haven't already, pull your model:
    echo    "%OLLAMA%" pull llama3.1:70b
)

echo.
echo ============================================================
echo  Setup complete.
echo  To run: python main.py
echo  Make sure Ollama is running first: ollama serve
echo ============================================================
