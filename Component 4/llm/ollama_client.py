"""Ollama client — final submission backend."""
from __future__ import annotations
import requests
from config import Config


class OllamaUnavailable(RuntimeError):
    pass


def chat(messages, model=None, url=None, timeout=120):
    model = model or Config.OLLAMA_MODEL
    url = url or Config.OLLAMA_URL
    try:
        r = requests.post(
            f"{url}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=timeout,
        )
    except requests.exceptions.RequestException as e:
        raise OllamaUnavailable(str(e)) from e
    if r.status_code != 200:
        raise OllamaUnavailable(f"HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    if "message" not in data or "content" not in data["message"]:
        raise OllamaUnavailable(f"Malformed response: {data}")
    return data["message"]["content"]