from __future__ import annotations

import os
from io import BytesIO

from minio import Minio
from minio.error import S3Error


class MinioStorageClient:
    def __init__(self) -> None:
        endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        use_ssl = os.getenv("MINIO_USE_SSL", "false").lower() == "true"
        self.bucket = os.getenv("MINIO_BUCKET", "resumes")
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=use_ssl)

    def ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def upload_bytes(self, object_name: str, payload: bytes, content_type: str = "application/pdf") -> str:
        self.ensure_bucket()
        self.client.put_object(
            bucket_name=self.bucket,
            object_name=object_name,
            data=BytesIO(payload),
            length=len(payload),
            content_type=content_type,
        )
        return f"minio://{self.bucket}/{object_name}"

    def delete_object(self, object_name: str) -> None:
        try:
            self.client.remove_object(self.bucket, object_name)
        except S3Error:
            return
