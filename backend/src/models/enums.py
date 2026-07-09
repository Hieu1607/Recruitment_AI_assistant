from enum import Enum


class UploadStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class ProfileStatus(str, Enum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    ARCHIVED = "archived"


class GraduationStatus(str, Enum):
    UNKNOWN = "unknown"
    STUDYING = "studying"
    FINAL_YEAR = "final_year"
    GRADUATED = "graduated"


class MatchRunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CandidateEvaluationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    OUTDATED = "outdated"


class ContentSource(str, Enum):
    AI_DRAFT = "ai_draft"
    TEMPLATE = "template"
    MANUAL = "manual"


class SentStatus(str, Enum):
    NOT_SENT = "not_sent"
    SENT = "sent"
    FAILED = "failed"


class UserStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"


class RoleName(str, Enum):
    ADMIN = "admin"
    RECRUITER = "recruiter"
    VIEWER = "viewer"
