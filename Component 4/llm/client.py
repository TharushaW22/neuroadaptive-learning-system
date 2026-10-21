"""Unified LLM client. Dispatches to Ollama / Groq / Gemini based on .env."""
from __future__ import annotations
from config import Config

from llm.ollama_client import chat as ollama_chat, OllamaUnavailable
try:
    from llm.groq_client import chat as groq_chat, GroqUnavailable
except ImportError:
    groq_chat, GroqUnavailable = None, None
try:
    from llm.gemini_client import chat as gemini_chat, GeminiUnavailable
except ImportError:
    gemini_chat, GeminiUnavailable = None, None


class LLMUnavailable(RuntimeError):
    pass


def _try(name, fn, messages, **kw):
    try:
        return fn(messages, **kw), name
    except Exception as e:
        return None, f"{name} failed: {e}"


def chat(messages, backend=None, model=None):
    order = []
    primary = (backend or Config.LLM_BACKEND).lower()
    order.append(primary)
    for b in ("ollama", "groq", "gemini"):
        if b not in order:
            order.append(b)

    errors = []
    for b in order:
        if b == "ollama":
            text, err = _try("Ollama", ollama_chat, messages,
                             model=model or Config.OLLAMA_MODEL,
                             url=Config.OLLAMA_URL)
        elif b == "groq":
            if not Config.GROQ_API_KEY:
                errors.append("Groq: no API key")
                continue
            text, err = _try("Groq", groq_chat, messages,
                             model=Config.GROQ_MODEL,
                             api_key=Config.GROQ_API_KEY)
        elif b == "gemini":
            if not Config.GEMINI_API_KEY:
                errors.append("Gemini: no API key")
                continue
            text, err = _try("Gemini", gemini_chat, messages,
                             model=Config.GEMINI_MODEL,
                             api_key=Config.GEMINI_API_KEY)
        else:
            continue
        if text is not None:
            return text, b
        errors.append(str(err))

    raise LLMUnavailable(" | ".join(errors))