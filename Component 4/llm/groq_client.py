"""Groq client — free-tier temporary fallback."""
from __future__ import annotations
from groq import Groq, APIError
from config import Config


class GroqUnavailable(RuntimeError):
    pass


def chat(messages, model=None, api_key=None, timeout=60):
    api_key = api_key or Config.GROQ_API_KEY
    model = model or Config.GROQ_MODEL
    if not api_key:
        raise GroqUnavailable("GROQ_API_KEY not set")
    client = Groq(api_key=api_key)
    try:
        completion = client.chat.completions.create(
            model=model, messages=messages,
            temperature=0.7, max_tokens=900, stream=False,
        )
    except APIError as e:
        raise GroqUnavailable(f"Groq API error: {e}") from e
    return completion.choices[0].message.content