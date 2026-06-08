from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.services.ai_agent.langgraph_trace import (
    _default_base_dir,
    format_exception_payload,
    serialize_for_json,
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class ResumeParseTraceLogger:
    def __init__(self, base_dir: Optional[Path | str] = None):
        self.base_dir = Path(base_dir) if base_dir is not None else _default_base_dir()
        self.root_dir = self.base_dir / "resume_parsing"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root_dir / "index.jsonl"

    def _trace_file_path(self, trace_id: str) -> Path:
        day_dir = self.root_dir / datetime.now(UTC).strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        return day_dir / f"{trace_id}.json"

    def _read_trace(self, trace_id: str) -> Dict[str, Any]:
        path = self._trace_file_path(trace_id)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {
            "trace_id": trace_id,
            "status": "in_progress",
            "created_at": _utc_now_iso(),
            "metadata": {},
            "input": None,
            "events": [],
            "llm_attempts": [],
            "result": None,
            "error": None,
        }

    def _write_trace(self, trace_id: str, payload: Dict[str, Any]) -> Path:
        path = self._trace_file_path(trace_id)
        path.write_text(
            json.dumps(serialize_for_json(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def _append_index(self, payload: Dict[str, Any]) -> None:
        with self.index_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(serialize_for_json(payload), ensure_ascii=False) + "\n")

    def start_trace(self, *, trace_id: str, metadata: Dict[str, Any], trace_input: Dict[str, Any]) -> Path:
        trace = self._read_trace(trace_id)
        trace["status"] = "in_progress"
        trace["started_at"] = _utc_now_iso()
        trace["metadata"] = serialize_for_json(metadata)
        trace["input"] = serialize_for_json(trace_input)
        return self._write_trace(trace_id, trace)

    def record_event(self, *, trace_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        trace = self._read_trace(trace_id)
        trace["events"].append(
            {
                "event_type": event_type,
                "timestamp": _utc_now_iso(),
                "payload": serialize_for_json(payload),
            }
        )
        self._write_trace(trace_id, trace)

    def update_metadata(self, *, trace_id: str, metadata: Dict[str, Any]) -> None:
        trace = self._read_trace(trace_id)
        trace["metadata"] = {
            **trace.get("metadata", {}),
            **serialize_for_json(metadata),
        }
        self._write_trace(trace_id, trace)

    def record_llm_attempt(self, *, trace_id: str, payload: Dict[str, Any]) -> None:
        trace = self._read_trace(trace_id)
        trace["llm_attempts"].append(serialize_for_json(payload))
        self._write_trace(trace_id, trace)

    def finalize_trace(
        self,
        *,
        trace_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[BaseException | Dict[str, Any]] = None,
    ) -> Path:
        trace = self._read_trace(trace_id)
        trace["status"] = status
        trace["finished_at"] = _utc_now_iso()
        if result is not None:
            trace["result"] = serialize_for_json(result)
        if error is not None:
            trace["error"] = (
                format_exception_payload(error)
                if isinstance(error, BaseException)
                else serialize_for_json(error)
            )
        path = self._write_trace(trace_id, trace)
        self._append_index(
            {
                "trace_id": trace_id,
                "status": status,
                "finished_at": trace["finished_at"],
                "resume_document_id": trace.get("metadata", {}).get("resume_document_id"),
                "file_name": trace.get("metadata", {}).get("file_name"),
                "extraction_mode": trace.get("metadata", {}).get("extraction_mode"),
                "trace_file": str(path),
                "error": trace.get("error"),
            }
        )
        return path


_default_logger: ResumeParseTraceLogger | None = None


def get_resume_parse_trace_logger() -> ResumeParseTraceLogger:
    global _default_logger
    desired_base_dir = _default_base_dir()
    if _default_logger is None or _default_logger.base_dir != desired_base_dir:
        _default_logger = ResumeParseTraceLogger(base_dir=desired_base_dir)
    return _default_logger
