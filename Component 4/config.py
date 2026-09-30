"""Central configuration loader. Reads .env and exposes typed settings."""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


class Config:
    LLM_BACKEND: str = os.getenv("LLM_BACKEND", "ollama").lower()
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    APP_TITLE: str = os.getenv("APP_TITLE", "MD-AP2L Adaptive Tutor")
    RESULTS_DIR: str = os.getenv("RESULTS_DIR", "results")
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def summary(cls) -> str:
        return (
            f"backend={cls.LLM_BACKEND} | "
            f"ollama={cls.OLLAMA_MODEL}@{cls.OLLAMA_URL} | "
            f"groq={'set' if cls.GROQ_API_KEY else 'unset'} | "
            f"gemini={'set' if cls.GEMINI_API_KEY else 'unset'}"
        )


if __name__ == "__main__":
    print(Config.summary())