from __future__ import annotations

import os

from alembic import context
from sqlalchemy import engine_from_config, pool

from src.models.auth import RoleAssignment, UserAccount  # noqa: F401
from src.models.candidate import CandidateProfile, ExtractionTrace, ResumeDocument  # noqa: F401
from src.models.engagement import (  # noqa: F401
    InterviewQuestionSet,
    OutreachMessage,
    ShortlistCollection,
    ShortlistItem,
)
from src.models.matching import JobDescription, MatchResult, MatchRun, QuerySession, QueryTurn  # noqa: F401
from src.repositories.db import Base


config = context.config


def _database_url() -> str:
    from_ini = config.get_main_option("sqlalchemy.url")
    if from_ini:
        return from_ini
    return os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/recruitment_ai",
    )


def run_migrations_offline() -> None:
    url = _database_url()
    context.configure(
        url=url,
        target_metadata=Base.metadata,
        literal_binds=True,
        compare_type=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=Base.metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
