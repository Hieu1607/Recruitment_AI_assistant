import sys
import os
from pathlib import Path
from types import SimpleNamespace

try:
    from celery import Celery
except ModuleNotFoundError:
    class _EagerTask:
        def __init__(self, fn, *, bind: bool = False, max_retries: int = 0):
            self._fn = fn
            self.bind = bind
            self.max_retries = max_retries
            self.request = SimpleNamespace(id=None, retries=0)
            self.__name__ = getattr(fn, "__name__", "task")
            self.__doc__ = getattr(fn, "__doc__", None)

        def run(self, *args, **kwargs):
            if self.bind:
                return self._fn(self, *args, **kwargs)
            return self._fn(*args, **kwargs)

        def delay(self, *args, **kwargs):
            return self.run(*args, **kwargs)

        def retry(self, exc=None):
            if exc is not None:
                raise exc
            raise RuntimeError("Task retry requested")

    class Celery:  # type: ignore[override]
        def __init__(self, *_args, **_kwargs):
            self.conf = {}

        def task(self, *args, **kwargs):
            def decorator(fn):
                return _EagerTask(
                    fn,
                    bind=kwargs.get("bind", False),
                    max_retries=kwargs.get("max_retries", 0),
                )

            return decorator

        def autodiscover_tasks(self, *_args, **_kwargs):
            return None

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "worker",
    broker=redis_url,
    backend=redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_default_queue="default",
    task_routes={
        "worker.tasks.process_resume": {"queue": "resume_parse"},
        "worker.tasks.evaluate_resume_batch": {"queue": "candidate_evaluation"},
        "worker.tasks.evaluate_candidate": {"queue": "candidate_evaluation"},
        "worker.tasks.send_outreach_email": {"queue": "default"},
        "worker.tasks.*": {"queue": "default"},
    },
    beat_schedule={
        "recover-pending-resume-batches": {
            "task": "worker.tasks.recover_pending_resume_batches",
            "schedule": 15.0,
        },
    },
    task_track_started=True,
)

celery_app.autodiscover_tasks(["worker"])
