import os
from celery import Celery

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "worker",
    broker=redis_url,
    backend=redis_url
)

celery_app.conf.task_routes = {
    "worker.tasks.*": {"queue": "default"}
}
