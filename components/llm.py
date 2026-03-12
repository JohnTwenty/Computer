"""
LLM client for Ollama's local inference server.
Streams tokens via the /api/chat endpoint.
"""

import json
from typing import Iterator

import requests


class LLMClient:
    def __init__(self, config):
        cfg = config["llm"]
        self.base_url = cfg.get("base_url", "http://localhost:11434").rstrip("/")
        self.model = cfg.get("model", "llama3.1:70b")
        self.system_prompt = cfg.get("system_prompt", "You are a helpful assistant.")
        print(f"LLM client configured: {self.model} @ {self.base_url}")

    def check_connection(self) -> bool:
        """Return True if Ollama is reachable and the configured model is available."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code != 200:
                return False
            models = [m["name"] for m in r.json().get("models", [])]
            # Ollama model names may include a tag like "llama3.1:70b" or just "llama3.1"
            base = self.model.split(":")[0]
            if not any(m.startswith(base) for m in models):
                print(f"WARNING: model '{self.model}' not found in Ollama.")
                print(f"  Available: {models}")
                print(f"  Run: ollama pull {self.model}")
            return True
        except requests.exceptions.ConnectionError:
            return False

    def stream_response(self, messages: list) -> Iterator[str]:
        """
        Yield text tokens as they stream from Ollama.
        `messages` is a list of {"role": ..., "content": ...} dicts.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                *messages,
            ],
            "stream": True,
        }
        url = f"{self.base_url}/api/chat"

        with requests.post(url, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                data = json.loads(raw_line)
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
                if data.get("done"):
                    break
