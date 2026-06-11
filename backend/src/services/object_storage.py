from __future__ import annotations

import uuid
from datetime import timedelta
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Optional
from urllib.parse import urlparse

from minio import Minio
from minio.error import S3Error

from src.core.config import settings


def format_storage_uri(bucket: str, object_key: str) -> str:
    normalized_key = object_key.lstrip("/")
    if not bucket or not normalized_key:
        raise ValueError("bucket and object_key are required")
    return f"s3://{bucket}/{normalized_key}"


def parse_storage_uri(storage_uri: str) -> tuple[str, str]:
    parsed = urlparse(storage_uri)
    object_key = parsed.path.lstrip("/")
    if parsed.scheme != "s3" or not parsed.netloc or not object_key:
        raise ValueError(f"Unsupported storage URI: {storage_uri}")
    return parsed.netloc, object_key


def build_object_key(
    *,
    prefix: str,
    original_filename: str,
    object_id: Optional[str] = None,
) -> str:
    safe_filename = Path(original_filename or "upload.pdf").name.strip() or "upload.pdf"
    safe_filename = safe_filename.replace("/", "_").replace("\\", "_")
    normalized_prefix = prefix.strip("/")
    unique_suffix = object_id or str(uuid.uuid4())
    if normalized_prefix:
        return f"{normalized_prefix}/{unique_suffix}_{safe_filename}"
    return f"{unique_suffix}_{safe_filename}"


class ObjectStorageService:
    def __init__(
        self,
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool,
        region: Optional[str] = None,
        default_bucket: Optional[str] = None,
        presigned_get_expiry_seconds: int = 3600,
    ) -> None:
        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region=region or None,
        )
        self._default_bucket = default_bucket
        self._region = region or None
        self._presigned_get_expiry_seconds = presigned_get_expiry_seconds
        self._ensured_buckets: set[str] = set()
        self._lock = Lock()

    def ensure_bucket_exists(self, bucket: Optional[str] = None) -> str:
        bucket_name = bucket or self._default_bucket
        if not bucket_name:
            raise ValueError("bucket is required")

        if bucket_name in self._ensured_buckets:
            return bucket_name

        with self._lock:
            if bucket_name in self._ensured_buckets:
                return bucket_name
            if not self._client.bucket_exists(bucket_name):
                kwargs = {"location": self._region} if self._region else {}
                self._client.make_bucket(bucket_name, **kwargs)
            self._ensured_buckets.add(bucket_name)

        return bucket_name

    def upload_bytes(
        self,
        *,
        data: bytes,
        object_key: str,
        content_type: str = "application/octet-stream",
        bucket: Optional[str] = None,
    ) -> str:
        bucket_name = self.ensure_bucket_exists(bucket)
        payload = BytesIO(data)
        self._client.put_object(
            bucket_name,
            object_key,
            data=payload,
            length=len(data),
            content_type=content_type,
        )
        return format_storage_uri(bucket_name, object_key)

    def download_bytes(self, storage_uri: str) -> bytes:
        bucket_name, object_key = parse_storage_uri(storage_uri)
        response = self._client.get_object(bucket_name, object_key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def delete_object(self, storage_uri: str) -> None:
        bucket_name, object_key = parse_storage_uri(storage_uri)
        try:
            self._client.remove_object(bucket_name, object_key)
        except S3Error as exc:
            if exc.code in {"NoSuchBucket", "NoSuchKey", "NoSuchObject"}:
                return
            raise

    def presigned_get_url(
        self,
        storage_uri: str,
        *,
        expires_seconds: Optional[int] = None,
    ) -> str:
        bucket_name, object_key = parse_storage_uri(storage_uri)
        expiry = expires_seconds or self._presigned_get_expiry_seconds
        return self._client.presigned_get_object(
            bucket_name,
            object_key,
            expires=timedelta(seconds=expiry),
        )

    def public_object_url(self, storage_uri: str) -> str:
        bucket_name, object_key = parse_storage_uri(storage_uri)
        if settings.MINIO_PUBLIC_BASE_URL:
            return f"{settings.MINIO_PUBLIC_BASE_URL.rstrip('/')}/{bucket_name}/{object_key}"
        scheme = "https" if settings.MINIO_SECURE else "http"
        return f"{scheme}://{settings.MINIO_ENDPOINT}/{bucket_name}/{object_key}"


@lru_cache(maxsize=1)
def get_object_storage() -> ObjectStorageService:
    return ObjectStorageService(
        endpoint=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
        region=settings.MINIO_REGION,
        default_bucket=settings.MINIO_RESUME_BUCKET,
        presigned_get_expiry_seconds=settings.MINIO_PRESIGNED_GET_EXPIRY_SECONDS,
    )
