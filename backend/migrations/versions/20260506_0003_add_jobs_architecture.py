"""Add jobs architecture and backfill job ownership.

Revision ID: 20260506_0003
Revises: 20260429_0002
Create Date: 2026-05-06 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid


revision = "20260506_0003"
down_revision = "20260429_0002"
branch_labels = None
depends_on = None

FALLBACK_USER_ID = "00000000-0000-0000-0000-000000000000"
FALLBACK_JOB_ID = "00000000-0000-0000-0000-000000000001"
FALLBACK_EMAIL = "system-migration@recruitment.local"
FALLBACK_NAME = "System Migration User"
FALLBACK_JOB_TITLE = "Legacy Imported Records"


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("owner_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["owner_user_id"], ["user_accounts.id"], name="fk_jobs_owner_user", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_jobs"),
    )
    op.create_index("ix_jobs_owner_user_id", "jobs", ["owner_user_id"], unique=False)

    op.add_column("job_descriptions", sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("resume_documents", sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.create_index("ix_job_descriptions_job_id", "job_descriptions", ["job_id"], unique=False)
    op.create_index("ix_resume_documents_job_id", "resume_documents", ["job_id"], unique=False)

    bind = op.get_bind()
    users = bind.execute(sa.text("SELECT id, display_name, email FROM user_accounts")).fetchall()
    existing_owner_ids = {
        row[0] for row in bind.execute(sa.text("SELECT owner_user_id FROM jobs")).fetchall()
    }
    for user_id, display_name, email in users:
        if user_id in existing_owner_ids:
            continue
        bind.execute(
            sa.text(
                """
                INSERT INTO jobs (id, owner_user_id, title, status)
                VALUES (:id, :owner_user_id, :title, 'active')
                """
            ),
            {
                "id": str(uuid.uuid4()),
                "owner_user_id": str(user_id),
                "title": f"Default Job - {display_name or email}",
            },
        )

    fallback_user_exists = bind.execute(
        sa.text("SELECT 1 FROM user_accounts WHERE id = :id"),
        {"id": FALLBACK_USER_ID},
    ).scalar()
    if not fallback_user_exists:
        bind.execute(
            sa.text(
                """
                INSERT INTO user_accounts (id, email, display_name, password_hash, status)
                VALUES (:id, :email, :display_name, NULL, 'active')
                """
            ),
            {
                "id": FALLBACK_USER_ID,
                "email": FALLBACK_EMAIL,
                "display_name": FALLBACK_NAME,
            },
        )

    fallback_job_exists = bind.execute(
        sa.text("SELECT 1 FROM jobs WHERE id = :id"),
        {"id": FALLBACK_JOB_ID},
    ).scalar()
    if not fallback_job_exists:
        bind.execute(
            sa.text(
                """
                INSERT INTO jobs (id, owner_user_id, title, status)
                VALUES (:id, :owner_user_id, :title, 'active')
                """
            ),
            {
                "id": FALLBACK_JOB_ID,
                "owner_user_id": FALLBACK_USER_ID,
                "title": FALLBACK_JOB_TITLE,
            },
        )

    op.execute(
        """
        UPDATE job_descriptions jd
        SET job_id = j.id
        FROM jobs j
        WHERE jd.job_id IS NULL
          AND jd.created_by_user_id = j.owner_user_id
        """
    )

    op.execute(
        """
        UPDATE resume_documents rd
        SET job_id = j.id
        FROM jobs j
        WHERE rd.job_id IS NULL
          AND rd.uploaded_by_user_id = j.owner_user_id
        """
    )

    bind.execute(
        sa.text(
            """
            UPDATE job_descriptions
            SET job_id = :fallback_job_id
            WHERE job_id IS NULL
            """
        ),
        {"fallback_job_id": FALLBACK_JOB_ID},
    )

    bind.execute(
        sa.text(
            """
            UPDATE resume_documents
            SET job_id = :fallback_job_id
            WHERE job_id IS NULL
            """
        ),
        {"fallback_job_id": FALLBACK_JOB_ID},
    )

    op.alter_column("job_descriptions", "job_id", nullable=False)
    op.alter_column("resume_documents", "job_id", nullable=False)

    op.create_foreign_key(
        "fk_job_descriptions_job_id_jobs",
        "job_descriptions",
        "jobs",
        ["job_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_resume_documents_job_id_jobs",
        "resume_documents",
        "jobs",
        ["job_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint("fk_resume_documents_job_id_jobs", "resume_documents", type_="foreignkey")
    op.drop_constraint("fk_job_descriptions_job_id_jobs", "job_descriptions", type_="foreignkey")
    op.drop_index("ix_resume_documents_job_id", table_name="resume_documents")
    op.drop_index("ix_job_descriptions_job_id", table_name="job_descriptions")
    op.drop_column("resume_documents", "job_id")
    op.drop_column("job_descriptions", "job_id")
    op.drop_index("ix_jobs_owner_user_id", table_name="jobs")
    op.drop_table("jobs")
