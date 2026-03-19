from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.services.observability.audit_logger import audit_log


@dataclass
class RetentionRunResult:
    started_at: datetime
    anonymized_records: int
    removed_objects: int


def run_retention_job() -> RetentionRunResult:
    result = RetentionRunResult(
        started_at=datetime.utcnow(),
        anonymized_records=0,
        removed_objects=0,
    )
    audit_log(
        "retention_job",
        {
            "started_at": result.started_at.isoformat(),
            "anonymized_records": result.anonymized_records,
            "removed_objects": result.removed_objects,
        },
    )
    return result
