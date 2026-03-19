from __future__ import annotations

import logging
import time

from worker.jobs.retention_job import run_retention_job


def run_worker_loop(poll_seconds: int = 300) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("recruitment.worker")

    logger.info("Worker started")
    while True:
        result = run_retention_job()
        logger.info(
            "Retention tick completed | started_at=%s anonymized=%s removed=%s",
            result.started_at.isoformat(),
            result.anonymized_records,
            result.removed_objects,
        )
        time.sleep(poll_seconds)


if __name__ == "__main__":
    run_worker_loop()
