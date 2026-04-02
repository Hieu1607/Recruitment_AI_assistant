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
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage

from src.prompts.build_prompts import build_prompts
from src.services.llm_service import LLMProvider

logger = logging.getLogger(__name__)

_llm = LLMProvider()

# All valid filterable/queryable fields on CandidateProfile (excludes id/full_name)
_ALL_CANDIDATE_FIELDS: frozenset = frozenset({
    "phone", "email", "location_normalized", "contact", "current_job_title",
    "educated", "ever_studied_abroad", "major", "cpa",
    "education_text", "experience_text", "experience_years", "skills_text",
    "languages_text", "projects_text", "summary_text", "achievements_text",
    "publications_text", "certifications_text", "references_text", "other_text",
})
_ALWAYS_INCLUDE: frozenset = frozenset({"id", "full_name"})

_MAX_CANDIDATES_FOR_RAG = 10


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
            if operator == "eq" and str(fv).lower() == str(value).lower():
                filtered.append(c)
            elif operator == "contains" and str(value).lower() in str(fv).lower():
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
            results = [c for c in results if keyword.lower() in str(c.get(field) or "").lower()]

    # Should clauses (OR contains) — keep any that match at least one
    should = dsl.get("should") or []
    if should:
        seen: set = set()
        matched: List[Dict] = []
        for clause in should:
            field, keyword = clause.get("field"), clause.get("contains", "")
            if not (field and keyword):
                continue
            for c in results:
                cid = str(c.get("id") or id(c))
                if cid not in seen and keyword.lower() in str(c.get(field) or "").lower():
                    matched.append(c)
                    seen.add(cid)
        results = matched if matched else results

    return results


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

    logger.info("[router_node] question=%r", question)

    prompt = build_prompts.build_router_prompt(question)
    response = _llm.generate(prompt)

    try:
        router_output = _parse_json(response.text)
    except Exception:
        logger.warning("[router_node] failed to parse JSON response, defaulting to LLM path")
        router_output = {
            "is_recruitment_related": True,
            "refusal_message": None,
            "relevant_fields": [],
            "dsl_question_query": None,
            "llm_question_query": question,
            "dsl_relevant_fields": [],
            "llm_relevant_fields": [],
            "reasoning": "Parse failure – fell back to LLM path",
        }

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
    dsl_question: str = router_output.get("dsl_question_query") or state.get("question") or ""
    dsl_relevant_fields: List[str] = router_output.get("dsl_relevant_fields") or []

    logger.info("[dsl_node] question=%r | fields=%s", dsl_question, dsl_relevant_fields)

    candidates = _fetch_candidates(dsl_relevant_fields)
    logger.info("[dsl_node] fetched %d candidate(s) from DB", len(candidates))

    prompt = build_prompts.build_dsl_query_prompt(dsl_question)
    response = _llm.generate(prompt)

    try:
        dsl = _parse_json(response.text)
        logger.info("[dsl_node] generated DSL filter: %s", json.dumps(dsl, ensure_ascii=False))
        dsl_candidates = _apply_dsl(candidates, dsl)
    except Exception:
        logger.warning("[dsl_node] failed to parse/apply DSL, returning all candidates")
        dsl_candidates = candidates

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
    llm_question: str = router_output.get("llm_question_query") or state.get("question") or ""
    llm_relevant_fields: List[str] = (
        router_output.get("llm_relevant_fields")
        or router_output.get("relevant_fields")
        or []
    )

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

    candidates = _fetch_candidates(llm_relevant_fields, candidate_ids)
    logger.info("[llm_node] fetched %d candidate(s) for LLM analysis", len(candidates))

    prompt = build_prompts.build_llm_query_prompt(llm_question, candidates)
    response = _llm.generate(prompt)

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
      3. Empty → return a "no match" message without calling the LLM
    """
    router_output: Dict = state.get("router_output") or {}
    question: str = state.get("question") or ""
    dsl_candidates: Optional[List[Dict]] = state.get("dsl_candidates")
    llm_result: Optional[Dict] = state.get("llm_result")

    logger.info("[answer_node] question=%r", question)

    # --- Determine final candidate IDs ---
    final_ids: Optional[List[str]] = None

    if llm_result:
        qualified = llm_result.get("qualified_candidates") or {}
        if isinstance(qualified, dict) and qualified:
            final_ids = list(qualified.keys())
            logger.info("[answer_node] source=llm | %d qualified candidate(s)", len(final_ids))
        else:
            final_ids = []
            logger.info("[answer_node] source=llm | no qualified candidates")
    elif dsl_candidates is not None:
        final_ids = [str(c["id"]) for c in dsl_candidates if c.get("id")]
        logger.info("[answer_node] source=dsl | %d candidate(s)", len(final_ids))
    else:
        logger.info("[answer_node] source=none | no filter ran")

    if final_ids is not None and len(final_ids) == 0:
        no_match = "No candidates matched the query."
        logger.info("[answer_node] result: no match")
        return {
            "messages": [AIMessage(content=no_match)],
            "answer": no_match,
        }

    # --- Collect all relevant fields from both stages ---
    all_relevant_fields: List[str] = list(set(
        (router_output.get("dsl_relevant_fields") or [])
        + (router_output.get("llm_relevant_fields") or router_output.get("relevant_fields") or [])
    ))
    logger.info("[answer_node] fetching fields=%s for ids=%s", all_relevant_fields,
                f"{len(final_ids)} IDs" if final_ids is not None else "all")

    # --- Fetch candidate data from DB ---
    candidates = _fetch_candidates(all_relevant_fields, final_ids)
    logger.info("[answer_node] fetched %d candidate(s) from DB", len(candidates))

    if not candidates:
        no_match = "No candidates matched the query."
        logger.info("[answer_node] result: no candidates after DB fetch")
        return {
            "messages": [AIMessage(content=no_match)],
            "answer": no_match,
        }

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
    prompt = build_prompts.build_answer_prompt(question, candidates)
    response = _llm.generate(prompt)

    answer = response.text.strip()
    logger.info("[answer_node] answer (first 200 chars): %r", answer[:200])
    return {
        "messages": [AIMessage(content=answer)],
        "answer": answer,
    }
