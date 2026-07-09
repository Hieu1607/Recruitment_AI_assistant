import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

module_path = Path(__file__).resolve().parents[1] / "worker" / "celery_app.py"
spec = importlib.util.spec_from_file_location("isolated_worker_celery_app", module_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
celery_app = module.celery_app


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
