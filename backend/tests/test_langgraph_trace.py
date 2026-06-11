import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.ai_agent.langgraph_trace import LangGraphTraceLogger, _default_base_dir  # noqa: E402


def test_langgraph_trace_logger_writes_request_trace_and_index(tmp_path):
    logger = LangGraphTraceLogger(base_dir=tmp_path)
    trace_id = "trace-test-001"

    logger.start_trace(
        trace_id=trace_id,
        metadata={
            "endpoint": "/api/v1/jobs/job-1/chat",
            "session_id": "session-1",
            "question": "Who should I interview?",
        },
        graph_input={
            "messages": [HumanMessage(content="Who should I interview?")],
            "question": "Who should I interview?",
        },
    )

    logger.record_node(
        trace_id=trace_id,
        node_name="router",
        state_before={"question": "Who should I interview?"},
        state_update={"router_output": {"llm_question_query": "Who should I interview?"}},
        state_after={
            "question": "Who should I interview?",
            "router_output": {"llm_question_query": "Who should I interview?"},
        },
        duration_ms=12.5,
        llm_call={
            "prompt": "router prompt",
            "response_text": "{\"llm_question_query\": \"Who should I interview?\"}",
            "response_raw": {"mock": True},
        },
    )

    logger.finalize_trace(
        trace_id=trace_id,
        status="success",
        graph_output={
            "answer": "Interview Candidate One first.",
            "messages": [AIMessage(content="Interview Candidate One first.")],
        },
    )

    trace_files = list(tmp_path.glob("langgraph/*/trace-test-001.json"))
    assert len(trace_files) == 1

    payload = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert payload["trace_id"] == trace_id
    assert payload["status"] == "success"
    assert payload["graph_input"]["messages"][0]["type"] == "human"
    assert payload["nodes"][0]["node_name"] == "router"
    assert payload["nodes"][0]["llm_call"]["response_text"] == "{\"llm_question_query\": \"Who should I interview?\"}"
    assert payload["graph_output"]["messages"][0]["content"] == "Interview Candidate One first."

    index_path = tmp_path / "langgraph" / "index.jsonl"
    lines = index_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    index_event = json.loads(lines[0])
    assert index_event["trace_id"] == trace_id
    assert index_event["status"] == "success"


def test_langgraph_trace_logger_serializes_common_runtime_objects(tmp_path):
    logger = LangGraphTraceLogger(base_dir=tmp_path)
    now = datetime.now(timezone.utc)
    trace_id = str(uuid4())

    logger.start_trace(
        trace_id=trace_id,
        metadata={"started_at": now},
        graph_input={
            "messages": [HumanMessage(content="Hello"), AIMessage(content="Hi")],
            "job_id": uuid4(),
            "path": Path("logs/example.json"),
        },
    )
    logger.finalize_trace(trace_id=trace_id, status="error", error={"message": "boom"})

    trace_file = next(tmp_path.glob(f"langgraph/*/{trace_id}.json"))
    payload = json.loads(trace_file.read_text(encoding="utf-8"))
    assert payload["metadata"]["started_at"] == now.isoformat()
    assert payload["graph_input"]["messages"][1]["type"] == "ai"
    assert isinstance(payload["graph_input"]["job_id"], str)
    assert payload["graph_input"]["path"] == "logs/example.json"


def test_default_trace_dir_uses_temp_location_under_pytest(monkeypatch):
    monkeypatch.delenv("LANGGRAPH_TRACE_LOG_DIR", raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "backend/tests/test_langgraph_trace.py::test_default_trace_dir")

    assert _default_base_dir() == Path(tempfile.gettempdir()) / "recruitment_ai_assistant_test_logs"
