"""LangGraph node implementations for the recruitment query chatbot.

Flow:
  trim → router → dsl? → llm? → answer

Prompts used (from BuildPrompts):
  - build_router_prompt   : decides DSL vs LLM vs both
  - build_dsl_query_prompt: translates question to structured JSON filter
  - build_llm_query_prompt: runs semantic analysis over candidate data
"""

import json
import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage

from src.core.config import settings
from src.prompts.build_prompts import build_prompts
from src.services.ai_agent.langgraph_trace import format_exception_payload, get_trace_logger
from src.services.llm_service import LLMProvider

logger = logging.getLogger(__name__)

_llm: LLMProvider | None = None
_AI_AGENT_LLM_MAX_TOKENS = 8192


def _get_llm() -> LLMProvider:
    global _llm
    if _llm is None:
        _llm = LLMProvider(max_tokens=max(settings.LLM_MAX_TOKENS, _AI_AGENT_LLM_MAX_TOKENS))
    return _llm


def _record_llm_trace(
    *,
    state: Dict[str, Any],
    node_name: str,
    prompt: str,
    response=None,
    error: Optional[BaseException] = None,
) -> None:
    trace_id = state.get("trace_id")
    if not trace_id:
        return

    payload: Dict[str, Any] = {
        "node_name": node_name,
        "prompt": prompt,
    }
    if response is not None:
        payload["response_text"] = response.text
        payload["response_provider"] = response.provider
        payload["response_model"] = response.model
        payload["response_usage"] = response.usage
        payload["response_raw"] = response.raw
    if error is not None:
        payload["error"] = format_exception_payload(error)

    get_trace_logger().record_event(
        trace_id=trace_id,
        event_type="llm_call",
        payload=payload,
    )

# All valid filterable/queryable fields on CandidateProfile (excludes id/full_name)
_ALL_CANDIDATE_FIELDS: frozenset = frozenset({
    "phone", "email", "location_normalized", "contact", "current_job_title",
    "graduation_status", "ever_studied_abroad", "major", "cpa",
    "education_text", "experience_text", "experience_years", "skills_text",
    "languages_text", "projects_text", "summary_text", "achievements_text",
    "publications_text", "certifications_text", "references_text", "other_text",
})
_ALWAYS_INCLUDE: frozenset = frozenset({"id", "full_name"})
_ALWAYS_SEMANTIC_FIELDS: frozenset = frozenset({"summary_text"})

_MAX_CANDIDATES_FOR_RAG = 10
_SEMANTIC_ONLY_DSL_FIELDS: frozenset = frozenset(
    {"contact", "current_job_title", "major", "cpa"}
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> Any:
    """Extract and parse the first JSON object/array from LLM output."""
    text = text.strip()
    # Strip markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.rstrip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def _default_router_output(question: str) -> Dict[str, Any]:
    return {
        "is_recruitment_related": True,
        "refusal_message": None,
        "relevant_fields": [],
        "dsl_question_query": None,
        "llm_question_query": question,
        "dsl_relevant_fields": [],
        "llm_relevant_fields": [],
        "reasoning": "Parse failure – fell back to LLM path",
    }


def _normalize_text_match(value: Any) -> str:
    text = str(value or "").strip().lower()
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _fetch_candidates(
    fields: List[str],
    candidate_ids: Optional[List[str]] = None,
) -> List[Dict]:
    """Query the DB fetching only the specified fields (plus id/full_name).

    Args:
        fields: Candidate fields to retrieve (validated against _ALL_CANDIDATE_FIELDS).
        candidate_ids: Optional list of candidate UUIDs to restrict the query to.

    Returns:
        List of candidate dicts containing only the requested fields.
    """
    from sqlalchemy.orm import load_only

    from src.models.candidate_profile import CandidateProfile
    from src.models.session import SessionLocal

    keep = list(_ALWAYS_INCLUDE | (set(fields) & _ALL_CANDIDATE_FIELDS))
    attrs = [getattr(CandidateProfile, f) for f in keep if hasattr(CandidateProfile, f)]

    db = SessionLocal()
    try:
        query = db.query(CandidateProfile).options(load_only(*attrs))
        if candidate_ids:
            query = query.filter(CandidateProfile.id.in_(candidate_ids))
        rows = query.all()
        result: List[Dict] = []
        for row in rows:
            d: Dict[str, Any] = {}
            for f in keep:
                val = getattr(row, f, None)
                if val is None:
                    d[f] = None
                elif f == "id":
                    d[f] = str(val)
                elif f == "experience_years":
                    try:
                        d[f] = float(val)
                    except (TypeError, ValueError):
                        d[f] = val
                else:
                    d[f] = val
            result.append(d)
        return result
    finally:
        db.close()


def _normalize_candidate_dict(candidate: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    keep = list(_ALWAYS_INCLUDE | (set(fields) & _ALL_CANDIDATE_FIELDS))
    normalized: Dict[str, Any] = {}
    for field in keep:
        value = candidate.get(field)
        if value is None:
            normalized[field] = None
        elif field == "id":
            normalized[field] = str(value)
        elif field == "experience_years":
            try:
                normalized[field] = float(value)
            except (TypeError, ValueError):
                normalized[field] = value
        else:
            normalized[field] = value
    return normalized


def _resolve_candidates(
    state: Dict[str, Any],
    fields: List[str],
    candidate_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    scoped_candidates = state.get("current_candidates")
    if scoped_candidates is None:
        return _fetch_candidates(fields, candidate_ids)

    allowed_ids = {str(candidate_id) for candidate_id in candidate_ids} if candidate_ids is not None else None
    result: List[Dict[str, Any]] = []
    for candidate in scoped_candidates:
        candidate_id = candidate.get("id")
        if allowed_ids is not None and str(candidate_id) not in allowed_ids:
            continue
        result.append(_normalize_candidate_dict(candidate, fields))
    return result


def _merge_semantic_fields(fields: List[str]) -> List[str]:
    return list(_ALWAYS_SEMANTIC_FIELDS | set(fields or []))


def _apply_dsl(candidates: List[Dict], dsl: Dict) -> List[Dict]:
    """Apply DSL filters/must/should clauses to a candidate list."""
    results = list(candidates)

    # Hard filters (AND)
    for field, condition in (dsl.get("filters") or {}).items():
        operator = condition.get("operator", "eq")
        value = condition.get("value")
        if value is None:
            continue
        filtered = []
        for c in results:
            fv = c.get(field)
            if fv is None:
                continue
            normalized_fv = _normalize_text_match(fv)
            normalized_value = _normalize_text_match(value)
            if operator == "eq" and normalized_fv == normalized_value:
                filtered.append(c)
            elif operator == "contains" and normalized_value in normalized_fv:
                filtered.append(c)
            elif operator == "gte":
                try:
                    if float(fv) >= float(value):
                        filtered.append(c)
                except (TypeError, ValueError):
                    pass
            elif operator == "lte":
                try:
                    if float(fv) <= float(value):
                        filtered.append(c)
                except (TypeError, ValueError):
                    pass
        results = filtered

    # Must clauses (AND contains)
    for clause in dsl.get("must") or []:
        field, keyword = clause.get("field"), clause.get("contains", "")
        if field and keyword:
            normalized_keyword = _normalize_text_match(keyword)
            results = [
                c
                for c in results
                if normalized_keyword in _normalize_text_match(c.get(field) or "")
            ]

    # Should clauses (OR contains) — keep any that match at least one
    should = dsl.get("should") or []
    if should:
        seen: set = set()
        matched: List[Dict] = []
        for clause in should:
            field, keyword = clause.get("field"), clause.get("contains", "")
            if not (field and keyword):
                continue
            normalized_keyword = _normalize_text_match(keyword)
            for c in results:
                cid = str(c.get("id") or id(c))
                if cid not in seen and normalized_keyword in _normalize_text_match(c.get(field) or ""):
                    matched.append(c)
                    seen.add(cid)
        results = matched if matched else results

    return results


def _match_candidates_by_name_in_question(
    candidates: List[Dict[str, Any]],
    question: str,
) -> List[Dict[str, Any]]:
    """Recover explicit name queries when generated DSL is too restrictive."""
    normalized_question = _normalize_text_match(question)
    if not normalized_question:
        return []

    matches: List[Dict[str, Any]] = []
    for candidate in candidates:
        normalized_name = _normalize_text_match(candidate.get("full_name"))
        if normalized_name and normalized_name in normalized_question:
            matches.append(candidate)
    return matches


def _sanitize_dsl_for_allowed_fields(
    dsl: Dict[str, Any],
    allowed_fields: List[str],
) -> Dict[str, Any]:
    allowed = set(allowed_fields or [])
    sanitized_filters = {
        field: condition
        for field, condition in (dsl.get("filters") or {}).items()
        if field in allowed
    }
    sanitized_must = [
        clause
        for clause in (dsl.get("must") or [])
        if clause.get("field") in allowed
    ]
    sanitized_should = [
        clause
        for clause in (dsl.get("should") or [])
        if clause.get("field") in allowed
    ]
    return {
        "filters": sanitized_filters,
        "must": sanitized_must,
        "should": sanitized_should,
    }


def _is_named_comparison_request(
    *,
    question: str,
    router_output: Dict[str, Any],
    dsl_candidates: Optional[List[Dict[str, Any]]],
) -> bool:
    if not dsl_candidates or len(dsl_candidates) < 2:
        return False
    dsl_fields = router_output.get("dsl_relevant_fields") or []
    if "full_name" not in dsl_fields:
        return False

    normalized_question = _normalize_text_match(question)
    comparison_markers = (
        "so sanh",
        "compare",
        "versus",
        "vs",
        "tot hon",
        "better fit",
        "better",
        "rank",
        "xep hang",
    )
    return any(marker in normalized_question for marker in comparison_markers)


def _question_uses_graduation_status_semantics(question: str) -> bool:
    normalized_question = _normalize_text_match(question)
    if not normalized_question:
        return False

    markers = (
        "chua tot nghiep",
        "chua ra truong",
        "dang hoc",
        "nam cuoi",
        "sinh vien nam cuoi",
        "final-year",
        "final year",
        "expected graduation",
        "du kien tot nghiep",
        "sap tot nghiep",
        "undergraduate student",
    )
    return any(marker in normalized_question for marker in markers)


def _override_router_output_for_graduation_status_semantics(
    *,
    question: str,
    router_output: Dict[str, Any],
) -> Dict[str, Any]:
    if not _question_uses_graduation_status_semantics(question):
        return router_output

    dsl_question = str(router_output.get("dsl_question_query") or "").strip().lower()
    dsl_fields = set(router_output.get("dsl_relevant_fields") or [])
    llm_fields = list(router_output.get("llm_relevant_fields") or [])

    educated_only_route = (
        dsl_question in {"educated = false", "graduation_status = studying", "graduation_status = final_year"}
        or dsl_fields in ({"educated"}, {"graduation_status"})
        or (router_output.get("relevant_fields") or []) in (["educated"], ["graduation_status"])
    )
    if not educated_only_route:
        return router_output

    updated = dict(router_output)
    updated["relevant_fields"] = ["education_text", "summary_text"]
    updated["dsl_question_query"] = None
    updated["dsl_relevant_fields"] = []
    updated["llm_question_query"] = question
    updated["llm_relevant_fields"] = ["education_text", "summary_text"]
    existing_reasoning = str(router_output.get("reasoning") or "").strip()
    suffix = (
        "Graduation-status semantics should use free-text education evidence rather than educated alone."
    )
    updated["reasoning"] = (
        f"{existing_reasoning} graduation-status semantics override: {suffix}"
        if existing_reasoning
        else f"graduation-status semantics override: {suffix}"
    )
    return updated


def _override_router_output_for_semantic_field_matching(
    *,
    question: str,
    router_output: Dict[str, Any],
) -> Dict[str, Any]:
    dsl_fields = list(router_output.get("dsl_relevant_fields") or [])
    semantic_dsl_fields = [
        field for field in dsl_fields if field in _SEMANTIC_ONLY_DSL_FIELDS
    ]
    if not semantic_dsl_fields:
        return router_output

    updated = dict(router_output)
    updated_dsl_fields = [
        field for field in dsl_fields if field not in _SEMANTIC_ONLY_DSL_FIELDS
    ]
    llm_fields = list(router_output.get("llm_relevant_fields") or [])
    for field in semantic_dsl_fields:
        if field not in llm_fields:
            llm_fields.append(field)
    updated["dsl_relevant_fields"] = updated_dsl_fields
    updated["llm_relevant_fields"] = llm_fields
    updated["relevant_fields"] = [
        field
        for field in (router_output.get("relevant_fields") or [])
        if field not in _SEMANTIC_ONLY_DSL_FIELDS
    ]
    if llm_fields and not updated.get("llm_question_query"):
        updated["llm_question_query"] = question
    if not updated_dsl_fields:
        updated["dsl_question_query"] = None

    existing_reasoning = str(router_output.get("reasoning") or "").strip()
    suffix = (
        "Fields such as current job title, major, CPA, and contact should use semantic LLM evidence because their values vary by language, formatting, and free-text conventions."
    )
    updated["reasoning"] = (
        f"{existing_reasoning} semantic-field override: {suffix}"
        if existing_reasoning
        else f"semantic-field override: {suffix}"
    )
    return updated


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Determine whether the question needs DSL, LLM, or both.

    Also acts as an off-topic guard: if the question is not recruitment-related,
    sets `answer` with a natural refusal so the graph can route directly to END.

    Uses build_router_prompt.
    """
    question: str = state.get("question") or ""
    job_context: Optional[Dict[str, Any]] = state.get("current_job")

    logger.info("[router_node] question=%r", question)

    prompt = build_prompts.build_router_prompt(question, job_context=job_context)
    try:
        response = _get_llm().generate(prompt)
    except Exception as exc:
        _record_llm_trace(state=state, node_name="router", prompt=prompt, error=exc)
        raise
    _record_llm_trace(state=state, node_name="router", prompt=prompt, response=response)

    try:
        router_output = _parse_json(response.text)
    except Exception:
        logger.warning("[router_node] failed to parse JSON response, defaulting to LLM path")
        router_output = _default_router_output(question)
    if not isinstance(router_output, dict):
        logger.warning(
            "[router_node] parsed JSON had unexpected type %s, defaulting to LLM path",
            type(router_output).__name__,
        )
        router_output = _default_router_output(question)

    router_output = _override_router_output_for_graduation_status_semantics(
        question=question,
        router_output=router_output,
    )
    router_output = _override_router_output_for_semantic_field_matching(
        question=question,
        router_output=router_output,
    )

    is_related: bool = bool(router_output.get("is_recruitment_related", True))

    if not is_related:
        refusal: str = (
            router_output.get("refusal_message")
            or "Xin lỗi, tôi chỉ có thể hỗ trợ các câu hỏi liên quan đến tuyển dụng và hồ sơ ứng viên."
        )
        logger.info("[router_node] off-topic → refusal: %r", refusal)
        return {
            "router_output": router_output,
            "answer": refusal,
            "messages": [AIMessage(content=refusal)],
        }

    logger.info(
        "[router_node] routing decision: dsl_query=%r | llm_query=%r | "
        "dsl_fields=%s | llm_fields=%s | reasoning=%r",
        router_output.get("dsl_question_query"),
        router_output.get("llm_question_query"),
        router_output.get("dsl_relevant_fields"),
        router_output.get("llm_relevant_fields"),
        router_output.get("reasoning"),
    )

    return {"router_output": router_output}


def dsl_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the DSL sub-question into structured filters and apply them.

    Fetches only the fields listed in router_output.dsl_relevant_fields from
    the database, then applies the generated DSL filter in memory.

    Uses build_dsl_query_prompt.
    """
    router_output: Dict = state.get("router_output") or {}
    question: str = state.get("question") or ""
    dsl_question: str = router_output.get("dsl_question_query") or state.get("question") or ""
    dsl_relevant_fields: List[str] = router_output.get("dsl_relevant_fields") or []

    logger.info("[dsl_node] question=%r | fields=%s", dsl_question, dsl_relevant_fields)

    candidates = _resolve_candidates(state, dsl_relevant_fields)
    logger.info("[dsl_node] fetched %d candidate(s) from DB", len(candidates))

    prompt = build_prompts.build_dsl_query_prompt(dsl_question)
    try:
        response = _get_llm().generate(prompt)
    except Exception as exc:
        _record_llm_trace(state=state, node_name="dsl", prompt=prompt, error=exc)
        raise
    _record_llm_trace(state=state, node_name="dsl", prompt=prompt, response=response)

    try:
        dsl = _parse_json(response.text)
        dsl = _sanitize_dsl_for_allowed_fields(dsl, dsl_relevant_fields)
        logger.info("[dsl_node] generated DSL filter: %s", json.dumps(dsl, ensure_ascii=False))
        dsl_candidates = _apply_dsl(candidates, dsl)
    except Exception:
        logger.warning("[dsl_node] failed to parse/apply DSL, returning all candidates")
        dsl_candidates = candidates

    if not dsl_candidates:
        fallback_candidates = _match_candidates_by_name_in_question(candidates, question)
        if fallback_candidates:
            logger.info(
                "[dsl_node] recovered %d candidate(s) by direct question-name matching fallback",
                len(fallback_candidates),
            )
            dsl_candidates = fallback_candidates

    logger.info(
        "[dsl_node] %d → %d candidate(s) after DSL filter",
        len(candidates),
        len(dsl_candidates),
    )

    return {"dsl_candidates": dsl_candidates}


def llm_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run semantic LLM analysis over the (optionally DSL-filtered) candidates.

    Fetches only the fields listed in router_output.llm_relevant_fields from
    the database. If the DSL node ran first, restricts the query to those
    candidate IDs.

    Uses build_llm_query_prompt.
    """
    router_output: Dict = state.get("router_output") or {}
    job_context: Optional[Dict[str, Any]] = state.get("current_job")
    llm_question: str = router_output.get("llm_question_query") or state.get("question") or ""
    llm_relevant_fields: List[str] = (
        router_output.get("llm_relevant_fields")
        or router_output.get("relevant_fields")
        or []
    )
    llm_relevant_fields = _merge_semantic_fields(llm_relevant_fields)

    # If DSL ran, restrict to its surviving candidate IDs
    dsl_candidates: Optional[List[Dict]] = state.get("dsl_candidates")
    candidate_ids: Optional[List[str]] = (
        [str(c["id"]) for c in dsl_candidates if c.get("id")]
        if dsl_candidates is not None
        else None
    )

    logger.info(
        "[llm_node] question=%r | fields=%s | candidate_ids_from_dsl=%s",
        llm_question,
        llm_relevant_fields,
        f"{len(candidate_ids)} IDs" if candidate_ids is not None else "all",
    )

    candidates = _resolve_candidates(state, llm_relevant_fields, candidate_ids)
    logger.info("[llm_node] fetched %d candidate(s) for LLM analysis", len(candidates))

    prompt = build_prompts.build_llm_query_prompt(llm_question, candidates, job_context=job_context)
    try:
        response = _get_llm().generate(prompt)
    except Exception as exc:
        _record_llm_trace(state=state, node_name="llm", prompt=prompt, error=exc)
        raise
    _record_llm_trace(state=state, node_name="llm", prompt=prompt, response=response)

    try:
        llm_result = _parse_json(response.text)
    except Exception:
        logger.warning("[llm_node] failed to parse LLM JSON response; raw=%r", response.text[:300])
        llm_result = {"total_qualified_candidates": 0, "qualified_candidates": {}}

    total = llm_result.get("total_qualified_candidates") or 0
    qualified_ids = list((llm_result.get("qualified_candidates") or {}).keys())
    logger.info(
        "[llm_node] qualified: %d candidate(s) → ids=%s",
        total,
        qualified_ids,
    )

    return {"llm_result": llm_result}


def answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """RAG answer node: fetch filtered candidates with all relevant fields and
    ask the LLM to produce a natural-language answer to the original question.

    Candidate pool resolution (in priority order):
      1. LLM qualified candidates (keyed by candidate_id in llm_result)
      2. DSL-filtered candidates
      3. Empty → ask the LLM to explain naturally that no candidates matched
    """
    router_output: Dict = state.get("router_output") or {}
    question: str = state.get("question") or ""
    job_context: Optional[Dict[str, Any]] = state.get("current_job")
    dsl_candidates: Optional[List[Dict]] = state.get("dsl_candidates")
    llm_result: Optional[Dict] = state.get("llm_result")
    named_comparison_request = _is_named_comparison_request(
        question=question,
        router_output=router_output,
        dsl_candidates=dsl_candidates,
    )

    logger.info("[answer_node] question=%r", question)

    # --- Determine final candidate IDs ---
    final_ids: Optional[List[str]] = None

    if named_comparison_request and dsl_candidates is not None:
        final_ids = [str(c["id"]) for c in dsl_candidates if c.get("id")]
        logger.info(
            "[answer_node] source=named_comparison_dsl | %d candidate(s)",
            len(final_ids),
        )
    elif llm_result:
        qualified = llm_result.get("qualified_candidates") or {}
        if isinstance(qualified, dict) and qualified:
            final_ids = list(qualified.keys())
            logger.info("[answer_node] source=llm | %d qualified candidate(s)", len(final_ids))
        elif dsl_candidates is not None:
            final_ids = [str(c["id"]) for c in dsl_candidates if c.get("id")]
            logger.info("[answer_node] source=dsl_fallback | %d candidate(s)", len(final_ids))
        else:
            final_ids = []
            logger.info("[answer_node] source=llm | no qualified candidates")
    elif dsl_candidates is not None:
        final_ids = [str(c["id"]) for c in dsl_candidates if c.get("id")]
        logger.info("[answer_node] source=dsl | %d candidate(s)", len(final_ids))
    else:
        logger.info("[answer_node] source=none | no filter ran")

    # --- Collect all relevant fields from both stages ---
    all_relevant_fields: List[str] = list(set(
        (router_output.get("dsl_relevant_fields") or [])
        + _merge_semantic_fields(router_output.get("llm_relevant_fields") or router_output.get("relevant_fields") or [])
    ))
    logger.info("[answer_node] fetching fields=%s for ids=%s", all_relevant_fields,
                f"{len(final_ids)} IDs" if final_ids is not None else "all")

    # --- Fetch candidate data from DB ---
    candidates = _resolve_candidates(state, all_relevant_fields, final_ids)
    logger.info("[answer_node] fetched %d candidate(s) from DB", len(candidates))

    if not candidates:
        logger.info("[answer_node] result: no candidates after DB fetch; asking LLM for natural no-match answer")

    # --- If too many candidates, trim to id + full_name only ---
    if len(candidates) > _MAX_CANDIDATES_FOR_RAG:
        logger.info(
            "[answer_node] %d candidates exceed limit (%d), trimming to id+full_name only",
            len(candidates),
            _MAX_CANDIDATES_FOR_RAG,
        )
        candidates = [{"id": c.get("id"), "full_name": c.get("full_name")} for c in candidates]

    # --- RAG: ask LLM to answer using the retrieved candidate data ---
    logger.info("[answer_node] calling LLM with %d candidate(s)", len(candidates))
    prompt = build_prompts.build_answer_prompt(question, candidates, job_context=job_context)
    try:
        response = _get_llm().generate(prompt)
    except Exception as exc:
        _record_llm_trace(state=state, node_name="answer", prompt=prompt, error=exc)
        raise
    _record_llm_trace(state=state, node_name="answer", prompt=prompt, response=response)

    answer = response.text.strip()
    logger.info("[answer_node] answer (first 200 chars): %r", answer[:200])
    return {
        "messages": [AIMessage(content=answer)],
        "answer": answer,
    }
