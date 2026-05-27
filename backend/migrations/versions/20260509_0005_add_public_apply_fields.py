"""Add public application link fields to jobs and candidate submitted fields.

Revision ID: 20260509_0005
Revises: 20260506_0004
Create Date: 2026-05-09 00:00:00
"""

from __future__ import annotations

import secrets

from alembic import op
import sqlalchemy as sa


revision = "20260509_0005"
down_revision = "20260506_0004"
branch_labels = None
depends_on = None


def _generate_token(existing_tokens: set[str]) -> str:
    while True:
        token = secrets.token_urlsafe(32)
        if token not in existing_tokens:
            existing_tokens.add(token)
            return token


def upgrade() -> None:
    op.add_column("jobs", sa.Column("public_apply_token", sa.String(length=64), nullable=True))
    op.add_column(
        "jobs",
        sa.Column("public_apply_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.add_column("jobs", sa.Column("candidate_message", sa.Text(), nullable=True))
    op.add_column(
        "jobs",
        sa.Column(
            "public_apply_created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.add_column("jobs", sa.Column("public_apply_disabled_at", sa.DateTime(timezone=True), nullable=True))

    op.add_column("candidate_profiles", sa.Column("submitted_full_name", sa.String(length=255), nullable=True))
    op.add_column("candidate_profiles", sa.Column("submitted_email", sa.String(length=320), nullable=True))

    bind = op.get_bind()
    rows = bind.execute(sa.text("SELECT id FROM jobs WHERE public_apply_token IS NULL")).fetchall()
    existing_tokens = {
        row[0]
        for row in bind.execute(
            sa.text("SELECT public_apply_token FROM jobs WHERE public_apply_token IS NOT NULL")
        ).fetchall()
    }
    for row in rows:
        bind.execute(
            sa.text("UPDATE jobs SET public_apply_token = :token WHERE id = :job_id"),
            {"token": _generate_token(existing_tokens), "job_id": row[0]},
        )

    op.alter_column("jobs", "public_apply_token", nullable=False)
    op.create_unique_constraint(op.f("uq_jobs_public_apply_token"), "jobs", ["public_apply_token"])


def downgrade() -> None:
    op.drop_constraint(op.f("uq_jobs_public_apply_token"), "jobs", type_="unique")
    op.drop_column("candidate_profiles", "submitted_email")
    op.drop_column("candidate_profiles", "submitted_full_name")
    op.drop_column("jobs", "public_apply_disabled_at")
    op.drop_column("jobs", "public_apply_created_at")
    op.drop_column("jobs", "candidate_message")
    op.drop_column("jobs", "public_apply_enabled")
    op.drop_column("jobs", "public_apply_token")
