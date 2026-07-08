from __future__ import annotations

import hashlib
import json
from typing import Any

SCORING_SIGNATURE_VERSION = "scoring-v1"
RUBRIC_PROMPT_VERSION = "rubric-prompt-v1"
SEMANTIC_PROMPT_VERSION = "semantic-prompt-v1"
MEASURABLE_RULE_VERSION = "measurable-rules-v1"


def compute_scoring_signature(
    *,
    job_description_id: Any,
    jd_text: str,
    hidden_text: str,
) -> str:
    payload = {
        "job_description_id": str(job_description_id),
        "jd_text": (jd_text or "").strip(),
        "hidden_text": (hidden_text or "").strip(),
        "rubric_prompt_version": RUBRIC_PROMPT_VERSION,
        "semantic_prompt_version": SEMANTIC_PROMPT_VERSION,
        "measurable_rule_version": MEASURABLE_RULE_VERSION,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{SCORING_SIGNATURE_VERSION}:{digest}"
