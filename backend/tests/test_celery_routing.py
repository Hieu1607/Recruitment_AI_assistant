import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.modules.pop("worker.tasks", None)
sys.modules.pop("worker.celery_app", None)
sys.modules.pop("worker", None)

celery_app = importlib.import_module("worker.celery_app").celery_app


def _config_value(name: str):
    config = celery_app.conf
    if isinstance(config, dict):
        return config[name]
    return getattr(config, name)


def test_resume_tasks_route_to_dedicated_queues():
    routes = _config_value("task_routes")

    assert routes["worker.tasks.process_resume"]["queue"] == "resume_parse"
    assert (
        routes["worker.tasks.evaluate_resume_batch"]["queue"]
        == "candidate_evaluation"
    )
    assert routes["worker.tasks.evaluate_candidate"]["queue"] == "candidate_evaluation"
    assert routes["worker.tasks.send_outreach_email"]["queue"] == "default"


def test_pending_batch_recovery_is_scheduled():
    beat_schedule = _config_value("beat_schedule")

    assert beat_schedule["recover-pending-resume-batches"] == {
        "task": "worker.tasks.recover_pending_resume_batches",
        "schedule": 15.0,
    }
