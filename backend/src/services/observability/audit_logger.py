from __future__ import annotations

import json
import logging
import re
from typing import Any


LOGGER = logging.getLogger("recruitment.audit")

EMAIL_PATTERN = re.compile(r"([A-Za-z0-9._%+-]{2})[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")
PHONE_PATTERN = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def _mask_text(value: str) -> str:
    value = EMAIL_PATTERN.sub(r"\1***\2", value)
    value = PHONE_PATTERN.sub("***masked-phone***", value)
    return value


def mask_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: mask_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [mask_payload(item) for item in payload]
    if isinstance(payload, str):
        return _mask_text(payload)
    return payload


def audit_log(event: str, payload: dict[str, Any]) -> None:
    masked = mask_payload(payload)
    LOGGER.info("%s | payload=%s", event, json.dumps(masked, ensure_ascii=True))
