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


def _candidates_for_llm(candidates: List[Dict], relevant_fields: List[str]) -> List[Dict]:
    """Project candidates to only the fields the LLM needs."""
    if not relevant_fields:
        return candidates
    id_fields = {"id", "full_name"}
    keep = id_fields | set(relevant_fields)
    return [{k: v for k, v in c.items() if k in keep} for c in candidates]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Determine whether the question needs DSL, LLM, or both.

    Uses build_router_prompt.
    """
    question: str = state.get("question") or ""
    current_candidates: List[Dict] = state.get("current_candidates") or []

    prompt = build_prompts.build_router_prompt(question, current_candidates)
    response = _llm.generate(prompt)

    try:
        router_output = _parse_json(response.text)
    except Exception:
        logger.warning("router_node: failed to parse JSON response, defaulting to LLM path")
        router_output = {
            "relevant_fields": [],
            "dsl_question_query": None,
            "llm_question_query": question,
            "dsl_relevant_fields": [],
            "llm_relevant_fields": [],
            "reasoning": "Parse failure – fell back to LLM path",
        }

    return {"router_output": router_output}


def dsl_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the DSL sub-question into structured filters and apply them.

    Uses build_dsl_query_prompt.
    """
    router_output: Dict = state.get("router_output") or {}
    dsl_question: str = router_output.get("dsl_question_query") or state.get("question") or ""
    current_candidates: List[Dict] = state.get("current_candidates") or []

    prompt = build_prompts.build_dsl_query_prompt(dsl_question, current_candidates)
    response = _llm.generate(prompt)

    try:
        dsl = _parse_json(response.text)
        dsl_candidates = _apply_dsl(current_candidates, dsl)
    except Exception:
        logger.warning("dsl_node: failed to parse/apply DSL, returning all candidates")
        dsl_candidates = current_candidates

    return {"dsl_candidates": dsl_candidates}


def llm_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run semantic LLM analysis over the (optionally DSL-filtered) candidates.

    Uses build_llm_query_prompt.
    NOTE: build_llm_query_prompt currently has a placeholder for candidate_data
    (always []). We work around this by appending serialised candidate data to
    the generated prompt ourselves.
    """
    router_output: Dict = state.get("router_output") or {}
    llm_question: str = router_output.get("llm_question_query") or state.get("question") or ""
    relevant_fields: List[str] = (
        router_output.get("llm_relevant_fields")
        or router_output.get("relevant_fields")
        or []
    )

    # Prefer DSL-filtered pool; fall back to full current list
    candidates: List[Dict] = state.get("dsl_candidates") or state.get("current_candidates") or []
    projected = _candidates_for_llm(candidates, relevant_fields)

    # build_llm_query_prompt has a placeholder (candidate_data=[]).
    # We obtain the base prompt then inject the real candidate data.
    base_prompt = build_prompts.build_llm_query_prompt(llm_question, projected, relevant_fields)
    # Replace the empty placeholder with real data
    real_data = json.dumps(projected, ensure_ascii=True, default=str)
    prompt = base_prompt.replace(
        "Candidate data: []",
        f"Candidate data: {real_data}",
    )

    response = _llm.generate(prompt)

    try:
        llm_result = _parse_json(response.text)
    except Exception:
        logger.warning("llm_node: failed to parse LLM JSON response")
        llm_result = {"total_qualified_candidates": 0, "qualified_candidates": {}}

    return {"llm_result": llm_result}


def answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Consolidate DSL and LLM results into a human-readable answer."""
    router_output: Dict = state.get("router_output") or {}
    dsl_candidates: Optional[List[Dict]] = state.get("dsl_candidates")
    llm_result: Optional[Dict] = state.get("llm_result")

    parts: List[str] = []

    if dsl_candidates is not None and router_output.get("dsl_question_query"):
        parts.append(f"Structured search found **{len(dsl_candidates)}** candidate(s).")

    if llm_result:
        total = llm_result.get("total_qualified_candidates") or 0
        qualified = llm_result.get("qualified_candidates") or {}
        if total:
            parts.append(f"LLM analysis identified **{total}** qualifying candidate(s):")
            items = qualified.items() if isinstance(qualified, dict) else []
            for cid, reason in items:
                parts.append(f"  - `{cid}`: {reason}")
        else:
            parts.append("LLM analysis found no candidates meeting the criteria.")

    if not parts:
        parts.append("No candidates matched the query.")

    answer = "\n".join(parts)
    return {
        "messages": [AIMessage(content=answer)],
        "answer": answer,
    }
