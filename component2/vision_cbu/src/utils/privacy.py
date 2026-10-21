"""Privacy helpers: pseudonymisation, consent-gated session saving, exports, deletion.

There is deliberately NO function in this project that writes raw video or raw frames.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import secrets
from datetime import datetime
from pathlib import Path

import pandas as pd


class ConsentRequiredError(PermissionError):
    pass


def _salt(cfg: dict) -> str:
    env = os.environ.get(cfg["privacy"].get("pseudonym_salt_env", "VCBU_PSEUDONYM_SALT"))
    if env:
        return env
    from src.utils.config import resolve_path
    p = resolve_path(cfg, "outputs_dir") / ".pseudonym_salt"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(secrets.token_hex(16), encoding="utf-8")
    return p.read_text(encoding="utf-8").strip()


def pseudonymize(identifier: str, salt: str) -> str:
    """Stable pseudonymous ID, e.g. 'P_3fa2c91b0d'. The salt must be kept private."""
    return "P_" + hashlib.sha256(f"{salt}:{identifier}".encode()).hexdigest()[:10]


def pseudonym_for(cfg: dict, identifier: str) -> str:
    return pseudonymize(identifier, _salt(cfg))


def save_session(df: pd.DataFrame, cfg: dict, consent: bool, session_label: str = "session") -> Path:
    """Save extracted FEATURES (never video). Requires logging enabled AND explicit consent."""
    if not cfg["privacy"].get("enable_session_logging", False):
        raise ConsentRequiredError("Session logging is disabled in settings.")
    if cfg["privacy"].get("require_consent", True) and not consent:
        raise ConsentRequiredError("Explicit consent is required before saving session data.")
    from src.utils.config import resolve_path
    out = resolve_path(cfg, "sessions_dir")
    out.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch for ch in session_label if ch.isalnum() or ch in "-_")[:40] or "session"
    path = out / f"{safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(path, index=False)
    return path


def delete_sessions(cfg: dict) -> int:
    from src.utils.config import resolve_path
    out = resolve_path(cfg, "sessions_dir")
    n = 0
    if out.exists():
        for f in out.glob("*.csv"):
            f.unlink()
            n += 1
    return n


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def to_json_bytes(obj, synthetic: bool | None = None, interpretation: dict | None = None) -> bytes:
    """JSON export. NaN/None become null. `interpretation` (see
    src.models.interpretation.export_metadata) labels heuristic score columns as experimental."""
    if isinstance(obj, pd.DataFrame):
        obj = json.loads(obj.to_json(orient="records"))
    payload = {"exported_at": datetime.now().isoformat(timespec="seconds"), "data": obj}
    if synthetic is not None:
        payload["is_synthetic"] = bool(synthetic)
        if synthetic:
            payload["notice"] = "SYNTHETIC / DEMO ONLY - not research data"
    if interpretation is not None:
        payload["heuristic_scores"] = interpretation
    return json.dumps(payload, indent=2, default=str).encode("utf-8")
