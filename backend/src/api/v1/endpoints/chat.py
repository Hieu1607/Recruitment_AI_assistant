"""Chatbot endpoint backed by the LangGraph recruitment query graph.

Session memory is held in-process (dict keyed by session_id). Each session
stores the last N messages; the graph's trim_node enforces the MEMORY_WINDOW.

Endpoints
---------
POST   /chat/          – send a message, get an answer
DELETE /chat/{session} – clear a chat session
GET    /chat/{session} – inspect session message history
"""

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.models.candidate_profile import CandidateProfile
from src.models.session import SessionLocal
from src.services.ai_agent.graph import get_graph

router = APIRouter()

# ---------------------------------------------------------------------------
# In-process session store  {session_id: [BaseMessage, ...]}
# ---------------------------------------------------------------------------
_sessions: Dict[str, List[Any]] = {}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User question")
    session_id: Optional[str] = Field(
        None,
        description="Session UUID. Omit to start a new session.",
    )
    candidate_limit: int = Field(
        500,
        ge=1,
        le=2000,
        description="Max number of candidate profiles loaded from DB per request.",
    )


class MessageItem(BaseModel):
    role: str
    content: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    candidates_in_scope: int


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: List[MessageItem]


class SessionDeleteResponse(BaseModel):
    session_id: str
    deleted: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_candidates(db, limit: int) -> List[Dict[str, Any]]:
    """Fetch CandidateProfile rows and serialise to plain dicts."""
    rows: List[CandidateProfile] = db.query(CandidateProfile).limit(limit).all()
    result = []
    for r in rows:
        result.append({
            "id": str(r.id),
            "full_name": r.full_name,
            "phone": r.phone,
            "email": r.email,
            "location_normalized": r.location_normalized,
            "contact": r.contact,
            "current_job_title": r.current_job_title,
            "educated": r.educated,
            "ever_studied_abroad": r.ever_studied_abroad,
            "major": r.major,
            "cpa": r.cpa,
            "education_text": r.education_text,
            "experience_text": r.experience_text,
            "experience_years": float(r.experience_years) if r.experience_years is not None else None,
            "skills_text": r.skills_text,
            "languages_text": r.languages_text,
            "projects_text": r.projects_text,
            "summary_text": r.summary_text,
            "achievements_text": r.achievements_text,
            "publications_text": r.publications_text,
            "certifications_text": r.certifications_text,
            "references_text": r.references_text,
            "other_text": r.other_text,
        })
    return result


def _messages_to_items(messages: List[Any]) -> List[MessageItem]:
    items = []
    for m in messages:
        role = getattr(m, "type", None) or "unknown"
        # LangChain message types: "human" | "ai" | "system"
        content = getattr(m, "content", str(m))
        items.append(MessageItem(role=role, content=content))
    return items


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/",
    response_model=ChatResponse,
    summary="Send a message to the recruitment chatbot",
    description=(
        "The chatbot answers questions about the candidate pool using a LangGraph "
        "pipeline (router → DSL filter → LLM analysis). Conversation history is "
        "kept per session (last 5 messages). Supply the returned `session_id` in "
        "subsequent requests to maintain context."
    ),
)
def chat(body: ChatRequest):
    # Resolve / create session
    session_id = body.session_id or str(uuid.uuid4())
    history: List[Any] = _sessions.get(session_id, [])

    # Load candidates from DB
    db = SessionLocal()
    try:
        candidates = _load_candidates(db, limit=body.candidate_limit)
    finally:
        db.close()

    # Append incoming human message to history
    human_msg = HumanMessage(content=body.message)
    history = list(history) + [human_msg]

    # Build graph input state
    graph_input = {
        "messages": history,
        "current_candidates": candidates,
        "question": body.message,
        "router_output": None,
        "dsl_candidates": None,
        "llm_result": None,
        "answer": "",
    }

    try:
        result = get_graph().invoke(graph_input)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph execution error: {exc}") from exc

    # Persist updated message history back to session store
    _sessions[session_id] = result.get("messages") or history

    answer: str = result.get("answer") or ""
    dsl_pool = result.get("dsl_candidates")
    candidates_in_scope = len(dsl_pool) if dsl_pool is not None else len(candidates)

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        candidates_in_scope=candidates_in_scope,
    )


@router.get(
    "/{session_id}",
    response_model=SessionHistoryResponse,
    summary="Get chat session history",
)
def get_session_history(session_id: str):
    history = _sessions.get(session_id)
    if history is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return SessionHistoryResponse(
        session_id=session_id,
        messages=_messages_to_items(history),
    )


@router.delete(
    "/{session_id}",
    response_model=SessionDeleteResponse,
    summary="Clear a chat session",
)
def delete_session(session_id: str):
    existed = session_id in _sessions
    _sessions.pop(session_id, None)
    if not existed:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return SessionDeleteResponse(session_id=session_id, deleted=True)
