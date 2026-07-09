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
import time
import unicodedata
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage

from src.core.config import settings
from src.prompts.build_prompts import build_prompts
from src.services.ai_agent.chat_batching import (
    AnswerMode,
    build_chat_map_batches,
    choose_answer_mode,
    compact_map_result,
    limit_compact_candidates,
)
from src.services.ai_agent.langgraph_trace import format_exception_payload, get_trace_logger
from src.services.llm_service import LLMProvider
from src.services.token_budget import BudgetWindow, estimate_json_tokens, estimate_tokens

logger = logging.getLogger(__name__)

_llm_cache: Dict[str, LLMProvider] = {}
_AI_AGENT_LLM_MAX_TOKENS = 8192
_CHAT_STAGE_MODEL_SETTINGS = {
    "router": "CHAT_ROUTER_MODEL_NAME",
    "dsl": "CHAT_DSL_MODEL_NAME",
    "map": "CHAT_MAP_MODEL_NAME",
    "reduce": "CHAT_REDUCE_MODEL_NAME",
    "answer": "CHAT_ANSWER_MODEL_NAME",
}


def _get_llm(stage: str = "default") -> LLMProvider:
    cached = _llm_cache.get(stage)
    if cached is not None:
        return cached

    model_setting = _CHAT_STAGE_MODEL_SETTINGS.get(stage)
    model_name = getattr(settings, model_setting, None) if model_setting else None
    llm = LLMProvider(
        model_name=model_name,
        max_tokens=max(settings.LLM_MAX_TOKENS, _AI_AGENT_LLM_MAX_TOKENS),
    )
    _llm_cache[stage] = llm
    return llm


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


def _record_chat_trace_event(
    *,
    state: Dict[str, Any],
    event_type: str,
    payload: Dict[str, Any],
) -> None:
    trace_id = state.get("trace_id")
    if not trace_id:
        return
    get_trace_logger().record_event(
        trace_id=trace_id,
        event_type=event_type,
        payload=payload,
    )


def _duration_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)

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
_DSL_SUPPORTED_FIELDS: frozenset = frozenset(
    {
        "full_name",
        "phone",
        "email",
        "location_normalized",
        "graduation_status",
        "ever_studied_abroad",
        "experience_years",
    }
)
_LLM_ONLY_DSL_FIELDS: frozenset = frozenset(
    {"contact", "current_job_title", "major", "cpa"} | (_ALL_CANDIDATE_FIELDS - (_DSL_SUPPORTED_FIELDS - {"full_name"}))
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
        "response_intent": "attribute_lookup",
        "relevant_fields": [],
        "dsl_question_query": None,
        "llm_question_query": question,
        "dsl_relevant_fields": [],
        "llm_relevant_fields": [],
        "reasoning": "Parse failure – fell back to LLM path",
    }


def _normalize_text_match(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("đ", "d")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _normalize_phone_match(value: Any) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def _normalize_email_match(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_filter_values(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _tokenize_normalized_text(value: str) -> List[str]:
    return [token for token in re.split(r"\W+", value) if token]


def _match_full_name_value(candidate_value: Any, query_value: Any) -> bool:
    normalized_candidate = _normalize_text_match(candidate_value)
    normalized_query = _normalize_text_match(query_value)
    if not normalized_candidate or not normalized_query:
        return False
    if " " in normalized_query:
        return normalized_query in normalized_candidate
    return normalized_query in _tokenize_normalized_text(normalized_candidate)


def _match_phone_value(candidate_value: Any, query_value: Any) -> bool:
    normalized_candidate = _normalize_phone_match(candidate_value)
    normalized_query = _normalize_phone_match(query_value)
    if not normalized_candidate or not normalized_query:
        return False
    return normalized_query in normalized_candidate


def _match_email_value(candidate_value: Any, query_value: Any) -> bool:
    normalized_candidate = _normalize_email_match(candidate_value)
    normalized_query = _normalize_email_match(query_value)
    if not normalized_candidate or not normalized_query:
        return False
    return normalized_query in normalized_candidate


def _match_eq_value(field: str, candidate_value: Any, query_value: Any) -> bool:
    if field == "full_name":
        return _match_full_name_value(candidate_value, query_value)
    if field == "phone":
        return _match_phone_value(candidate_value, query_value)
    if field == "email":
        return _match_email_value(candidate_value, query_value)
    return _normalize_text_match(candidate_value) == _normalize_text_match(query_value)


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


def _inventory_answer_language(question: str) -> str:
    normalized = (question or "").strip().lower()
    if re.search(r"\b(which|what|who|show|list|count|candidate|candidates)\b", normalized):
        return "en"
    return "vi"


def _render_inventory_answer(question: str, candidates: List[Dict[str, Any]]) -> str:
    language = _inventory_answer_language(question)
    names = [str(candidate.get("full_name") or "").strip() for candidate in candidates]
    names = [name for name in names if name]
    total = len(names)

    if language == "en":
        if total == 0:
            return (
                "There are currently 0 candidates in scope.\n\n"
                "Would you like me to narrow the pool by skills, experience, or education?"
            )
        lines = [f"There are currently {total} candidates in scope:", ""]
        lines.extend(f"- {name}" for name in names)
        lines.extend(
            [
                "",
                f"Total: {total} candidates.",
                "",
                "Would you like me to narrow the pool by skills, experience, or education?",
            ]
        )
        return "\n".join(lines)

    if total == 0:
        return (
            "Hiện có 0 ứng viên trong phạm vi hiện tại.\n\n"
            "Bạn muốn mình lọc tiếp theo kỹ năng, kinh nghiệm hay học vấn?"
        )

    lines = [f"Hiện có {total} ứng viên trong phạm vi hiện tại:", ""]
    lines.extend(f"- {name}" for name in names)
    lines.extend(
        [
            "",
            f"Tổng cộng: {total} ứng viên.",
            "",
            "Bạn muốn mình lọc tiếp theo kỹ năng, kinh nghiệm hay học vấn?",
        ]
    )
    return "\n".join(lines)


def _apply_dsl(candidates: List[Dict], dsl: Dict) -> List[Dict]:
    """Apply DSL filters/must/should clauses to a candidate list."""
    results = list(candidates)

    # Hard filters (AND)
    for field, condition in (dsl.get("filters") or {}).items():
        operator = condition.get("operator", "eq")
        value = condition.get("value")
        if value is None:
            continue
        values = _normalize_filter_values(value)
        filtered = []
        for c in results:
            fv = c.get(field)
            if fv is None:
                continue
            normalized_fv = _normalize_text_match(fv)
            normalized_value = _normalize_text_match(value)
            if operator == "eq" and any(_match_eq_value(field, fv, item) for item in values):
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
        full_name = candidate.get("full_name")
        normalized_name = _normalize_text_match(full_name)
        if normalized_name and normalized_name in normalized_question:
            matches.append(candidate)
            continue
        candidate_tokens = _tokenize_normalized_text(normalized_name)
        if any(token and re.search(rf"\b{re.escape(token)}\b", normalized_question) for token in candidate_tokens):
            matches.append(candidate)
    return matches


def _normalize_map_response(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"qualifiedCandidates": [], "batchQualifiedCount": 0}
    qualified = payload.get("qualifiedCandidates") or payload.get("qualified_candidates") or []
    if isinstance(qualified, dict):
        qualified = [
            {"id": candidate_id, "reason": reason}
            for candidate_id, reason in qualified.items()
        ]
    if not isinstance(qualified, list):
        qualified = []
    return compact_map_result(
        {
            "qualifiedCandidates": qualified,
            "batchQualifiedCount": int(payload.get("batchQualifiedCount") or len(qualified)),
        }
    )


def _normalize_reduce_response(payload: Any, map_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    ranked = (
        payload.get("rankedCandidates")
        or payload.get("ranked_candidates")
        or payload.get("qualifiedCandidates")
        or []
    )
    if isinstance(ranked, dict):
        ranked = [
            {"id": candidate_id, "reason": reason}
            for candidate_id, reason in ranked.items()
        ]
    if not isinstance(ranked, list):
        ranked = []
    if not ranked:
        for result in map_results:
            ranked.extend(result.get("qualifiedCandidates") or [])

    normalized_candidates: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for candidate in ranked:
        if not isinstance(candidate, dict) or not candidate.get("id"):
            continue
        candidate_id = str(candidate["id"])
        if candidate_id in seen_ids:
            continue
        seen_ids.add(candidate_id)
        normalized_candidates.append(
            {
                "id": candidate_id,
                "name": candidate.get("name") or candidate.get("full_name"),
                "score": candidate.get("score", 0),
                "reason": candidate.get("reason") or "",
            }
        )

    total = int(
        payload.get("totalQualified")
        or payload.get("total_qualified_candidates")
        or len(normalized_candidates)
    )
    return {
        "total_qualified_candidates": total,
        "qualified_candidates": {
            candidate["id"]: candidate.get("reason") or ""
            for candidate in normalized_candidates
        },
        "ranked_candidates": normalized_candidates,
    }


def _reduce_single_map_batch(map_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not map_results:
        return _normalize_reduce_response({}, map_results)
    first = map_results[0] or {}
    qualified_candidates = first.get("qualifiedCandidates") or []
    return _normalize_reduce_response(
        {
            "rankedCandidates": qualified_candidates,
            "totalQualified": first.get("batchQualifiedCount") or len(qualified_candidates),
        },
        map_results,
    )


def _sanitize_dsl_for_allowed_fields(
    dsl: Dict[str, Any],
    allowed_fields: List[str],
) -> Dict[str, Any]:
    allowed = set(allowed_fields or []) & set(_DSL_SUPPORTED_FIELDS)
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
        field for field in dsl_fields if field in _LLM_ONLY_DSL_FIELDS
    ]
    if not semantic_dsl_fields:
        return router_output

    updated = dict(router_output)
    updated_dsl_fields = [
        field for field in dsl_fields if field not in _LLM_ONLY_DSL_FIELDS
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
        if field not in _LLM_ONLY_DSL_FIELDS
    ]
    if llm_fields and not updated.get("llm_question_query"):
        updated["llm_question_query"] = question
    if not updated_dsl_fields:
        updated["dsl_question_query"] = None

    existing_reasoning = str(router_output.get("reasoning") or "").strip()
    suffix = (
        "Fields such as contact, current job title, major, CPA, and free-text profile fields should use semantic LLM evidence instead of DSL filtering."
    )
    updated["reasoning"] = (
        f"{existing_reasoning} semantic-field override: {suffix}"
        if existing_reasoning
        else f"semantic-field override: {suffix}"
    )
    return updated


def _question_targets_school_entity(question: str) -> bool:
    normalized_question = _normalize_text_match(question)
    if not normalized_question:
        return False

    school_markers = (
        "dai hoc",
        "truong",
        "university",
        "college",
        "hoc vien",
        "institute of technology",
    )
    education_intent_markers = (
        "dang hoc",
        "hoc tai",
        "tot nghiep",
        "graduated from",
        "study at",
        "studied at",
        "alumni",
    )
    return any(marker in normalized_question for marker in school_markers) and any(
        marker in normalized_question for marker in education_intent_markers
    )


def _override_router_output_for_school_entity_queries(
    *,
    question: str,
    router_output: Dict[str, Any],
) -> Dict[str, Any]:
    if not _question_targets_school_entity(question):
        return router_output

    dsl_fields = list(router_output.get("dsl_relevant_fields") or [])
    if "location_normalized" not in dsl_fields:
        return router_output

    updated = dict(router_output)
    updated_dsl_fields = [field for field in dsl_fields if field != "location_normalized"]
    llm_fields = list(router_output.get("llm_relevant_fields") or [])
    for field in ("education_text", "summary_text"):
        if field not in llm_fields:
            llm_fields.append(field)

    updated["dsl_relevant_fields"] = updated_dsl_fields
    updated["llm_relevant_fields"] = llm_fields
    updated["relevant_fields"] = [
        field
        for field in (router_output.get("relevant_fields") or [])
        if field != "location_normalized"
    ]
    if llm_fields and not updated.get("llm_question_query"):
        updated["llm_question_query"] = question
    if not updated_dsl_fields:
        updated["dsl_question_query"] = None

    existing_reasoning = str(router_output.get("reasoning") or "").strip()
    suffix = (
        "School and university questions should use education evidence, not location_normalized."
    )
    updated["reasoning"] = (
        f"{existing_reasoning} school-entity override: {suffix}"
        if existing_reasoning
        else f"school-entity override: {suffix}"
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
        response = _get_llm("router").generate(prompt)
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
    router_output = _override_router_output_for_school_entity_queries(
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
    dsl_relevant_fields: List[str] = [
        field for field in (router_output.get("dsl_relevant_fields") or []) if field in _DSL_SUPPORTED_FIELDS
    ]

    logger.info("[dsl_node] question=%r | fields=%s", dsl_question, dsl_relevant_fields)

    candidates = _resolve_candidates(state, dsl_relevant_fields)
    logger.info("[dsl_node] fetched %d candidate(s) from DB", len(candidates))

    prompt = build_prompts.build_dsl_query_prompt(dsl_question)
    try:
        response = _get_llm("dsl").generate(prompt)
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

    chat_window = BudgetWindow(
        context_window=settings.CHAT_CONTEXT_WINDOW_TOKENS,
        output_budget=settings.CHAT_OUTPUT_TOKEN_BUDGET,
        reserve=settings.CHAT_CONTEXT_RESERVE_TOKENS,
    )
    static_prompt_tokens = estimate_tokens(llm_question) + estimate_json_tokens(job_context or {})
    map_batches = build_chat_map_batches(
        question=llm_question,
        candidates=candidates,
        job_context=job_context,
        static_prompt_tokens=static_prompt_tokens,
        window=chat_window,
        max_candidates_per_batch=settings.CHAT_MAX_CANDIDATES_PER_MAP_BATCH,
    )
    _record_chat_trace_event(
        state=state,
        event_type="chat_map_plan_created",
        payload={
            "candidateCount": len(candidates),
            "mapBatchCount": len(map_batches),
            "plannerSettings": {
                "staticPromptTokens": static_prompt_tokens,
                "inputBudgetTokens": chat_window.input_budget,
                "outputBudgetTokens": chat_window.output_budget,
                "contextWindowTokens": chat_window.context_window,
                "reserveTokens": chat_window.reserve,
                "maxCandidatesPerMapBatch": settings.CHAT_MAX_CANDIDATES_PER_MAP_BATCH,
            },
            "batchSizes": [len(batch.candidates) for batch in map_batches],
        },
    )

    map_results: List[Dict[str, Any]] = []
    for batch_index, map_batch in enumerate(map_batches):
        map_started_at = time.perf_counter()
        _record_chat_trace_event(
            state=state,
            event_type="chat_map_batch_started",
            payload={
                "batchIndex": batch_index,
                "candidateCount": len(map_batch.candidates),
                "estimatedInputTokens": map_batch.estimated_input_tokens,
                "estimatedOutputTokens": map_batch.estimated_output_tokens,
            },
        )
        prompt = build_prompts.build_chat_semantic_map_prompt(
            llm_question,
            map_batch.candidates,
            job_context=job_context,
        )
        try:
            response = _get_llm("map").generate(prompt)
        except Exception as exc:
            _record_llm_trace(state=state, node_name="llm_map", prompt=prompt, error=exc)
            raise
        _record_llm_trace(state=state, node_name="llm_map", prompt=prompt, response=response)

        try:
            map_result = _normalize_map_response(_parse_json(response.text))
        except Exception:
            logger.warning("[llm_node] failed to parse map JSON response; raw=%r", response.text[:300])
            map_result = {"qualifiedCandidates": [], "batchQualifiedCount": 0}
        map_results.append(map_result)
        _record_chat_trace_event(
            state=state,
            event_type="chat_map_batch_completed",
            payload={
                "batchIndex": batch_index,
                "durationMs": _duration_ms(map_started_at),
                "qualifiedCandidateCount": len(map_result.get("qualifiedCandidates") or []),
            },
        )

    reduce_started_at = time.perf_counter()
    if len(map_results) <= 1:
        llm_result = _reduce_single_map_batch(map_results)
        _record_chat_trace_event(
            state=state,
            event_type="chat_reduce_skipped",
            payload={
                "reason": "single_map_batch",
                "mapResultCount": len(map_results),
            },
        )
    else:
        reduce_prompt = build_prompts.build_chat_reduce_prompt(
            llm_question,
            map_results,
            job_context=job_context,
        )
        try:
            reduce_response = _get_llm("reduce").generate(reduce_prompt)
        except Exception as exc:
            _record_llm_trace(state=state, node_name="llm_reduce", prompt=reduce_prompt, error=exc)
            raise
        _record_llm_trace(state=state, node_name="llm_reduce", prompt=reduce_prompt, response=reduce_response)

        try:
            llm_result = _normalize_reduce_response(_parse_json(reduce_response.text), map_results)
        except Exception:
            logger.warning("[llm_node] failed to parse reduce JSON response; raw=%r", reduce_response.text[:300])
            llm_result = _normalize_reduce_response({}, map_results)

    ranked_candidates = llm_result.get("ranked_candidates") or []
    answer_mode = choose_answer_mode(
        candidates=ranked_candidates,
        detailed_threshold=settings.CHAT_MAX_DETAILED_FINAL_CANDIDATES,
        compact_threshold=settings.CHAT_MAX_COMPACT_FINAL_CANDIDATES,
        estimated_full_tokens=estimate_json_tokens(ranked_candidates),
        final_input_budget=chat_window.input_budget,
    )
    llm_result["answer_mode"] = answer_mode.value
    _record_chat_trace_event(
        state=state,
        event_type="chat_reduce_completed",
        payload={
            "durationMs": _duration_ms(reduce_started_at),
            "mapResultCount": len(map_results),
            "qualifiedCandidateCount": len(llm_result.get("qualified_candidates") or {}),
            "answerMode": answer_mode.value,
        },
    )

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
    response_intent: str = str(router_output.get("response_intent") or "attribute_lookup")
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

    if response_intent == "inventory_list":
        final_ids = None
        logger.info("[answer_node] source=inventory_scope | using all candidates in current scope")
    elif named_comparison_request and dsl_candidates is not None:
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
    if response_intent == "inventory_list":
        all_relevant_fields = ["current_job_title", "summary_text"]
    logger.info("[answer_node] fetching fields=%s for ids=%s", all_relevant_fields,
                f"{len(final_ids)} IDs" if final_ids is not None else "all")

    # --- Fetch candidate data from DB ---
    candidates = _resolve_candidates(state, all_relevant_fields, final_ids)
    logger.info("[answer_node] fetched %d candidate(s) from DB", len(candidates))

    if not candidates:
        logger.info("[answer_node] result: no candidates after DB fetch; asking LLM for natural no-match answer")

    try:
        answer_mode = AnswerMode((llm_result or {}).get("answer_mode") or AnswerMode.DETAILED.value)
    except ValueError:
        answer_mode = AnswerMode.DETAILED

    # --- If too many candidates, trim to id + full_name only ---
    use_compact_answer = (
        response_intent != "inventory_list"
        and (answer_mode == AnswerMode.COMPACT_ID_NAME or len(candidates) > _MAX_CANDIDATES_FOR_RAG)
    )
    total_qualified = int((llm_result or {}).get("total_qualified_candidates") or len(candidates))
    if use_compact_answer:
        logger.info(
            "[answer_node] compact answer mode selected for %d candidate(s); trimming to id+full_name only",
            len(candidates),
        )
        max_compact = settings.CHAT_MAX_COMPACT_FINAL_CANDIDATES
        compact_candidates = limit_compact_candidates(candidates, max_compact)
        omitted_count = max(0, total_qualified - len(compact_candidates))
        candidates = compact_candidates
    else:
        omitted_count = 0

    # --- RAG: ask LLM to answer using the retrieved candidate data ---
    logger.info("[answer_node] calling LLM with %d candidate(s)", len(candidates))
    if response_intent == "inventory_list":
        answer = _render_inventory_answer(question, candidates)
        _record_chat_trace_event(
            state=state,
            event_type="chat_answer_prompt_built",
            payload={
                "answerMode": "inventory_list",
                "candidateCount": len(candidates),
                "totalQualifiedCandidates": len(candidates),
                "omittedCandidateCount": 0,
                "estimatedPromptTokens": 0,
                "renderedDeterministically": True,
            },
        )
        logger.info("[answer_node] rendered deterministic inventory answer (first 200 chars): %r", answer[:200])
        return {
            "messages": [AIMessage(content=answer)],
            "answer": answer,
        }
    elif use_compact_answer:
        prompt = build_prompts.build_compact_answer_prompt(
            question,
            candidates,
            total_count=total_qualified,
            omitted_count=omitted_count,
            job_context=job_context,
        )
    else:
        prompt = build_prompts.build_answer_prompt(question, candidates, job_context=job_context)
    if response_intent != "inventory_list":
        _record_chat_trace_event(
            state=state,
            event_type="chat_answer_prompt_built",
            payload={
                "answerMode": answer_mode.value,
                "candidateCount": len(candidates),
                "totalQualifiedCandidates": total_qualified,
                "omittedCandidateCount": omitted_count,
                "estimatedPromptTokens": estimate_tokens(prompt),
            },
        )
    try:
        response = _get_llm("answer").generate(prompt)
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
