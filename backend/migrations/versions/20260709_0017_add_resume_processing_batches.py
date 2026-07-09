"""Add durable resume processing batches.

Revision ID: 20260709_0017
Revises: 20260708_0016
Create Date: 2026-07-09 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260709_0017"
down_revision = "20260708_0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    batch_status = postgresql.ENUM(
        "parsing",
        "evaluation_pending",
        "evaluating",
        "completed",
        "completed_with_errors",
        "failed",
        name="resume_processing_batch_status_enum",
        create_type=False,
    )
    batch_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "resume_processing_batches",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("total_count", sa.Integer(), nullable=False),
        sa.Column("terminal_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("processed_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("failed_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("status", batch_status, server_default="parsing", nullable=False),
        sa.Column("evaluation_task_id", sa.String(length=255), nullable=True),
        sa.Column("evaluation_dispatch_attempted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("failed_count >= 0", name="ck_resume_batches_failed_nonnegative"),
        sa.CheckConstraint("processed_count >= 0", name="ck_resume_batches_processed_nonnegative"),
        sa.CheckConstraint("terminal_count >= 0", name="ck_resume_batches_terminal_nonnegative"),
        sa.CheckConstraint("total_count >= 1", name="ck_resume_batches_total_positive"),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_resume_processing_batches_job_id"),
        "resume_processing_batches",
        ["job_id"],
        unique=False,
    )
    op.add_column(
        "resume_documents",
        sa.Column("processing_batch_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_resume_documents_processing_batch_id",
        "resume_documents",
        "resume_processing_batches",
        ["processing_batch_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        op.f("ix_resume_documents_processing_batch_id"),
        "resume_documents",
        ["processing_batch_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_resume_documents_processing_batch_id"), table_name="resume_documents")
    op.drop_constraint(
        "fk_resume_documents_processing_batch_id",
        "resume_documents",
        type_="foreignkey",
    )
    op.drop_column("resume_documents", "processing_batch_id")
    op.drop_index(op.f("ix_resume_processing_batches_job_id"), table_name="resume_processing_batches")
    op.drop_table("resume_processing_batches")
    postgresql.ENUM(name="resume_processing_batch_status_enum").drop(
        op.get_bind(),
        checkfirst=True,
    )
