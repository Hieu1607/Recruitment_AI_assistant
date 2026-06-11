"""LangGraph workflow for the recruitment query chatbot.

State fields
------------
messages          : conversation history (last 5 kept via trim node)
current_candidates: candidates currently in scope for the session
question          : latest user question
router_output     : routing decision from router_node
dsl_candidates    : candidates surviving the DSL filter
llm_result        : parsed output from llm_node
answer            : final plain-text answer

Graph topology
--------------
START → trim → router ─┬─► dsl ─┬─► llm ─► answer → END
                        │        └─────────► answer → END
                        ├─────────────────► llm ──── answer → END
                        ├─────────────────────────── answer → END
                        └── (off-topic) ──────────────────── END
"""

from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.services.ai_agent.langgraph_trace import (
    format_exception_payload,
    get_trace_logger,
    merge_state_for_trace,
)
from src.services.ai_agent.nodes import answer_node, dsl_node, llm_node, router_node

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

MEMORY_WINDOW = 5  # number of recent messages to keep


class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    current_candidates: List[Dict[str, Any]]
    current_job: Optional[Dict[str, Any]]
    question: str
    router_output: Optional[Dict[str, Any]]
    dsl_candidates: Optional[List[Dict[str, Any]]]
    llm_result: Optional[Dict[str, Any]]
    answer: str
    trace_id: Optional[str]
    trace_metadata: Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Utility nodes
# ---------------------------------------------------------------------------

def trim_node(state: GraphState) -> Dict[str, Any]:
    """Retain only the last MEMORY_WINDOW messages to bound context size."""
    messages = state.get("messages") or []
    if len(messages) > MEMORY_WINDOW:
        messages = messages[-MEMORY_WINDOW:]
    return {"messages": messages}


def _with_trace(node_name: str, handler):
    def wrapped(state: GraphState) -> Dict[str, Any]:
        trace_id = state.get("trace_id")
        if not trace_id:
            return handler(state)

        started_at = datetime.utcnow()
        try:
            state_update = handler(state)
            duration_ms = (datetime.utcnow() - started_at).total_seconds() * 1000
            get_trace_logger().record_node(
                trace_id=trace_id,
                node_name=node_name,
                state_before=state,
                state_update=state_update,
                state_after=merge_state_for_trace(state, state_update),
                duration_ms=duration_ms,
            )
            return state_update
        except Exception as exc:
            duration_ms = (datetime.utcnow() - started_at).total_seconds() * 1000
            get_trace_logger().record_node(
                trace_id=trace_id,
                node_name=node_name,
                state_before=state,
                state_update={},
                state_after=state,
                duration_ms=duration_ms,
                error=format_exception_payload(exc),
            )
            raise

    return wrapped


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _route_after_router(state: GraphState) -> str:
    router_output = state.get("router_output") or {}
    # Off-topic: answer already populated by router_node, skip all processing
    if not router_output.get("is_recruitment_related", True):
        if state.get("trace_id"):
            get_trace_logger().record_event(
                trace_id=state["trace_id"],
                event_type="route_decision",
                payload={
                    "from_node": "router",
                    "to_node": END,
                    "reason": "question_not_recruitment_related",
                    "router_output": router_output,
                },
            )
        return END
    has_dsl = bool(router_output.get("dsl_question_query"))
    has_llm = bool(router_output.get("llm_question_query"))
    next_node = "answer"
    if has_dsl:
        next_node = "dsl"
    elif has_llm:
        next_node = "llm"
    if state.get("trace_id"):
        get_trace_logger().record_event(
            trace_id=state["trace_id"],
            event_type="route_decision",
            payload={
                "from_node": "router",
                "to_node": next_node,
                "router_output": router_output,
            },
        )
    return next_node


def _route_after_dsl(state: GraphState) -> str:
    router_output = state.get("router_output") or {}
    next_node = "llm" if bool(router_output.get("llm_question_query")) else "answer"
    if state.get("trace_id"):
        get_trace_logger().record_event(
            trace_id=state["trace_id"],
            event_type="route_decision",
            payload={
                "from_node": "dsl",
                "to_node": next_node,
                "router_output": router_output,
            },
        )
    return next_node


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(GraphState)

    g.add_node("trim", _with_trace("trim", trim_node))
    g.add_node("router", _with_trace("router", router_node))
    g.add_node("dsl", _with_trace("dsl", dsl_node))
    g.add_node("llm", _with_trace("llm", llm_node))
    g.add_node("answer", _with_trace("answer", answer_node))

    g.add_edge(START, "trim")
    g.add_edge("trim", "router")

    g.add_conditional_edges(
        "router",
        _route_after_router,
        {"dsl": "dsl", "llm": "llm", "answer": "answer", END: END},
    )
    g.add_conditional_edges(
        "dsl",
        _route_after_dsl,
        {"llm": "llm", "answer": "answer"},
    )

    g.add_edge("llm", "answer")
    g.add_edge("answer", END)

    return g.compile()


# Module-level compiled graph (import and use directly)
graph = build_graph()


def get_graph() -> StateGraph:
    return graph
