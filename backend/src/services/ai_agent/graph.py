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

from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.services.ai_agent.nodes import answer_node, dsl_node, llm_node, router_node

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

MEMORY_WINDOW = 5  # number of recent messages to keep


class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    current_candidates: List[Dict[str, Any]]
    question: str
    router_output: Optional[Dict[str, Any]]
    dsl_candidates: Optional[List[Dict[str, Any]]]
    llm_result: Optional[Dict[str, Any]]
    answer: str


# ---------------------------------------------------------------------------
# Utility nodes
# ---------------------------------------------------------------------------

def trim_node(state: GraphState) -> Dict[str, Any]:
    """Retain only the last MEMORY_WINDOW messages to bound context size."""
    messages = state.get("messages") or []
    if len(messages) > MEMORY_WINDOW:
        messages = messages[-MEMORY_WINDOW:]
    return {"messages": messages}


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _route_after_router(state: GraphState) -> str:
    router_output = state.get("router_output") or {}
    # Off-topic: answer already populated by router_node, skip all processing
    if not router_output.get("is_recruitment_related", True):
        return END
    has_dsl = bool(router_output.get("dsl_question_query"))
    has_llm = bool(router_output.get("llm_question_query"))
    if has_dsl:
        return "dsl"
    if has_llm:
        return "llm"
    return "answer"


def _route_after_dsl(state: GraphState) -> str:
    router_output = state.get("router_output") or {}
    return "llm" if bool(router_output.get("llm_question_query")) else "answer"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(GraphState)

    g.add_node("trim", trim_node)
    g.add_node("router", router_node)
    g.add_node("dsl", dsl_node)
    g.add_node("llm", llm_node)
    g.add_node("answer", answer_node)

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
