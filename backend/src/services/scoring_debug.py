from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

from src.services.ai_agent.langgraph_trace import _default_base_dir, format_exception_payload, serialize_for_json


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def preview_text(text: str | None, limit: int = 1200) -> dict[str, Any]:
    normalized = str(text or "")
    truncated = normalized[:limit]
    return {
        "length": len(normalized),
        "truncated": len(normalized) > limit,
        "text": truncated,
    }


def _normalize_payload(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    return serialize_for_json(value)


class ScoringDebugLogger:
    def __init__(self, match_run_id: str, base_dir: Optional[Path | str] = None):
        self.match_run_id = str(match_run_id)
        self.base_dir = Path(base_dir) if base_dir is not None else _default_base_dir()
        self.root_dir = self.base_dir / "scoring" / datetime.now(UTC).strftime("%Y-%m-%d")
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.root_dir / f"{self.match_run_id}.jsonl"

    def record_event(self, event: str, payload: dict[str, Any]) -> None:
        line = {
            "timestamp": _utc_now_iso(),
            "match_run_id": self.match_run_id,
            "event": event,
            "payload": _normalize_payload(payload),
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")

    def record_error(self, event: str, exc: BaseException, payload: Optional[dict[str, Any]] = None) -> None:
        merged = dict(payload or {})
        merged["error"] = format_exception_payload(exc)
        self.record_event(event, merged)
