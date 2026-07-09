from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


_CHARS_PER_TOKEN = 3.2


@dataclass(frozen=True)
class BudgetWindow:
    context_window: int
    output_budget: int
    reserve: int

    @property
    def input_budget(self) -> int:
        return max(0, int(self.context_window) - int(self.output_budget) - int(self.reserve))


def estimate_tokens(text: str | None) -> int:
    normalized = str(text or "")
    if not normalized:
        return 0
    return max(1, math.ceil(len(normalized) / _CHARS_PER_TOKEN))


def estimate_json_tokens(payload: Any) -> int:
    return estimate_tokens(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def fits_budget(estimated_input_tokens: int, window: BudgetWindow) -> bool:
    return estimated_input_tokens <= window.input_budget
