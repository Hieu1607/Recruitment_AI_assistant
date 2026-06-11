from __future__ import annotations

import json
import os
import tempfile
import traceback
from copy import deepcopy
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID

from langchain_core.messages import BaseMessage


def _default_base_dir() -> Path:
    configured = os.getenv("LANGGRAPH_TRACE_LOG_DIR")
    if configured:
        return Path(configured)
    if os.getenv("PYTEST_CURRENT_TEST"):
        return Path(tempfile.gettempdir()) / "recruitment_ai_assistant_test_logs"
    return Path(__file__).resolve().parents[4] / "logs"


def _serialize_message(message: Any) -> Dict[str, Any]:
    message_type = getattr(message, "type", None)
    if not message_type:
        class_name = message.__class__.__name__.lower()
        if class_name.endswith("message"):
            message_type = class_name.removesuffix("message")
        else:
            message_type = message.__class__.__name__
    return {
        "type": message_type,
        "content": getattr(message, "content", ""),
        "additional_kwargs": getattr(message, "additional_kwargs", {}),
        "response_metadata": getattr(message, "response_metadata", {}),
    }


def _is_message_like(value: Any) -> bool:
    if isinstance(value, BaseMessage):
        return True
    return hasattr(value, "content") and value.__class__.__name__.lower().endswith("message")


def serialize_for_json(value: Any) -> Any:
    if _is_message_like(value):
        return _serialize_message(value)
    if isinstance(value, dict):
        return {str(key): serialize_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_for_json(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, BaseException):
        return {
            "type": value.__class__.__name__,
            "message": str(value),
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def format_exception_payload(exc: BaseException) -> Dict[str, Any]:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "stack_trace": "".join(traceback.format_exception(exc)),
    }


def merge_state_for_trace(state_before: Dict[str, Any], state_update: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(state_before)
    for key, value in state_update.items():
        merged[key] = value
    return merged


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class LangGraphTraceLogger:
    def __init__(self, base_dir: Optional[Path | str] = None):
        self.base_dir = Path(base_dir) if base_dir is not None else _default_base_dir()
        self.root_dir = self.base_dir / "langgraph"
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
            "graph_input": None,
            "graph_output": None,
            "nodes": [],
            "events": [],
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

    def start_trace(self, *, trace_id: str, metadata: Dict[str, Any], graph_input: Dict[str, Any]) -> Path:
        payload = self._read_trace(trace_id)
        payload["trace_id"] = trace_id
        payload["status"] = "in_progress"
        payload["metadata"] = serialize_for_json(metadata)
        payload["graph_input"] = serialize_for_json(graph_input)
        payload["started_at"] = _utc_now_iso()
        return self._write_trace(trace_id, payload)

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

    def record_node(
        self,
        *,
        trace_id: str,
        node_name: str,
        state_before: Dict[str, Any],
        state_update: Dict[str, Any],
        state_after: Dict[str, Any],
        duration_ms: float,
        llm_call: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        trace = self._read_trace(trace_id)
        trace["nodes"].append(
            {
                "node_name": node_name,
                "timestamp": _utc_now_iso(),
                "duration_ms": duration_ms,
                "state_before": serialize_for_json(state_before),
                "state_update": serialize_for_json(state_update),
                "state_after": serialize_for_json(state_after),
                "llm_call": serialize_for_json(llm_call),
                "error": serialize_for_json(error),
            }
        )
        self._write_trace(trace_id, trace)

    def finalize_trace(
        self,
        *,
        trace_id: str,
        status: str,
        graph_output: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> Path:
        trace = self._read_trace(trace_id)
        trace["status"] = status
        trace["finished_at"] = _utc_now_iso()
        if graph_output is not None:
            trace["graph_output"] = serialize_for_json(graph_output)
        if error is not None:
            trace["error"] = serialize_for_json(error)
        path = self._write_trace(trace_id, trace)
        self._append_index(
            {
                "trace_id": trace_id,
                "status": status,
                "finished_at": trace["finished_at"],
                "endpoint": trace.get("metadata", {}).get("endpoint"),
                "session_id": trace.get("metadata", {}).get("session_id"),
                "job_id": trace.get("metadata", {}).get("job_id"),
                "question": trace.get("metadata", {}).get("question"),
                "trace_file": str(path),
                "error": trace.get("error"),
            }
        )
        return path


_default_logger: LangGraphTraceLogger | None = None


def get_trace_logger() -> LangGraphTraceLogger:
    global _default_logger
    desired_base_dir = _default_base_dir()
    if _default_logger is None or _default_logger.base_dir != desired_base_dir:
        _default_logger = LangGraphTraceLogger(base_dir=desired_base_dir)
    return _default_logger
