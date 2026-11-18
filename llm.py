"""Tenký klient pre Ollama REST API (embeddingy + chat). Všetko beží lokálne."""
import json
from typing import Iterator

import requests

import config


class OllamaClient:
    def __init__(self, host: str = config.OLLAMA_HOST, llm_model: str = config.LLM_MODEL,
                 embed_model: str = config.EMBED_MODEL, timeout: int = 600):
        self.host = host.rstrip("/")
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.timeout = timeout

    # --- embeddingy -------------------------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Prevedie zoznam textov na vektory. Ten istý model sa používa pre PDF aj otázky."""
        r = requests.post(f"{self.host}/api/embed",
                          json={"model": self.embed_model, "input": texts},
                          timeout=self.timeout)
        r.raise_for_status()
        return r.json()["embeddings"]

    # --- generovanie ------------------------------------------------------
    def chat(self, messages: list[dict], temperature: float = 0.1) -> str:
        r = requests.post(f"{self.host}/api/chat",
                          json={"model": self.llm_model, "messages": messages, "stream": False,
                                "options": {"temperature": temperature}},
                          timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def chat_stream(self, messages: list[dict], temperature: float = 0.1) -> Iterator[str]:
        with requests.post(f"{self.host}/api/chat",
                           json={"model": self.llm_model, "messages": messages, "stream": True,
                                 "options": {"temperature": temperature}},
                           stream=True, timeout=self.timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                yield data.get("message", {}).get("content", "")
                if data.get("done"):
                    break

    # --- diagnostika ------------------------------------------------------
    def available_models(self) -> list[str]:
        r = requests.get(f"{self.host}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def check(self) -> tuple[bool, str]:
        """Overí, že Ollama beží a potrebné modely sú stiahnuté."""
        try:
            models = self.available_models()
        except Exception as e:  # noqa: BLE001
            return False, f"Ollama nebeží na {self.host} ({e}). Spusti `ollama serve`."
        missing = [m for m in (self.llm_model, self.embed_model)
                   if not any(x == m or x.startswith(m + ":") or x.split(":")[0] == m for x in models)]
        if missing:
            return False, "Chýbajú modely: " + ", ".join(missing) + \
                   ". Stiahni ich: " + " a ".join(f"`ollama pull {m}`" for m in missing)
        return True, f"Ollama OK – LLM: {self.llm_model}, embeddingy: {self.embed_model}"
