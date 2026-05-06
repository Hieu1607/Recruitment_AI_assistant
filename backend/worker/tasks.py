from worker.celery_app import celery_app

@celery_app.task(name="worker.tasks.process_document")
def process_document(document_id: int):
    # Heavy NLP processing task
    print(f"Processing document {document_id}")
    return True

@celery_app.task(name="worker.tasks.cleanup_logs")
def cleanup_logs():
    # Regular cron job to cleanup logs
    print("Cleaning up logs")
    return True
