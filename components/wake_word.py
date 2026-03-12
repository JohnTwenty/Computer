"""
Wake word detection using openWakeWord.
Loads ONNX models directly so tflite-runtime is not required.
Runs on CPU, keeping the GPU free for Whisper and the LLM.
"""

import os
import pathlib

import numpy as np
import openwakeword
from openwakeword.model import Model


def _onnx_path(model_name: str) -> str:
    """
    Resolve the absolute path to an ONNX wake word model.
    Search order:
      1. models/ directory in the project root  (custom / third-party models)
      2. openWakeWord package resources/models  (built-in downloaded models)
    """
    # 1. Project models/ directory (relative to this file's parent)
    project_models = os.path.join(pathlib.Path(__file__).parent.parent, "models")
    local = os.path.join(project_models, f"{model_name}.onnx")
    if os.path.exists(local):
        return local

    # 2. openWakeWord package resources
    pkg_models = os.path.join(
        pathlib.Path(openwakeword.__file__).parent, "resources", "models"
    )
    bundled = os.path.join(pkg_models, f"{model_name}.onnx")
    if os.path.exists(bundled):
        return bundled

    raise FileNotFoundError(
        f"ONNX model '{model_name}' not found.\n"
        f"  Checked: {local}\n"
        f"           {bundled}\n"
        f"  To download a built-in model:\n"
        f"    python -c \"from openwakeword.utils import download_models; "
        f"download_models(['{model_name}'])\"\n"
        f"  Or place a custom .onnx file in the models/ directory."
    )


class WakeWordDetector:
    def __init__(self, config):
        cfg = config["wake_word"]
        self.threshold = cfg.get("threshold", 0.5)
        self.trigger_level = cfg.get("trigger_level", 1)  # consecutive chunks required
        self._hits = 0  # current consecutive high-score chunk count
        model_name = cfg.get("model", "hey_jarvis_v0.1")

        # Pass the full .onnx path so openwakeword uses onnxruntime, not tflite.
        onnx_path = _onnx_path(model_name)
        self.model = Model(
            wakeword_models=[onnx_path],
            enable_speex_noise_suppression=False,
        )
        # The prediction dict key is the basename without extension
        self.model_key = os.path.splitext(os.path.basename(onnx_path))[0]
        self.label = model_name.split("_v")[0].replace("_", " ").title()
        print(f"Wake word detector ready: '{self.label}' "
              f"(threshold={self.threshold}, trigger_level={self.trigger_level})")

    def process(self, audio_chunk: np.ndarray) -> bool:
        """
        Feed one int16 audio chunk through the model.
        Returns True only when the score has exceeded the threshold for
        `trigger_level` consecutive chunks — debounces single-frame noise spikes.
        Resets the counter on any miss.
        """
        predictions = self.model.predict(audio_chunk)
        score = predictions.get(self.model_key, 0.0)

        if float(score) >= self.threshold:
            self._hits += 1
        else:
            self._hits = 0

        if self._hits >= self.trigger_level:
            self._hits = 0  # reset so it doesn't immediately re-fire
            return True
        return False
