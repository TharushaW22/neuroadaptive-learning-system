"""Gemini client — free-tier temporary fallback."""
from __future__ import annotations
import requests
from config import Config

DEFAULT_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiUnavailable(RuntimeError):
    pass


def chat(messages, model=None, api_key=None, timeout=60):
    api_key = api_key or Config.GEMINI_API_KEY
    model = model or Config.GEMINI_MODEL
    if not api_key:
        raise GeminiUnavailable("GEMINI_API_KEY not set")

    system_text, contents = "", []
    for m in messages:
        if m["role"] == "system":
            system_text += m["content"] + "\n"
        elif m["role"] == "user":
            contents.append({"role": "user", "parts": [{"text": m["content"]}]})
        elif m["role"] == "assistant":
            contents.append({"role": "model", "parts": [{"text": m["content"]}]})

    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_text}]} if system_text else None,
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 900},
    }
    url = f"{DEFAULT_URL}/{model}:generateContent?key={api_key}"
    try:
        r = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.RequestException as e:
        raise GeminiUnavailable(str(e)) from e
    if r.status_code != 200:
        raise GeminiUnavailable(f"HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise GeminiUnavailable(f"Malformed response: {data}") from e