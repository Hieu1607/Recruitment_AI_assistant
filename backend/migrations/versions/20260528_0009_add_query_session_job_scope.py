"""Add job scope to query sessions.

Revision ID: 20260528_0009
Revises: 20260526_0008
Create Date: 2026-05-28 00:09:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260528_0009"
down_revision = "20260526_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "query_sessions",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_query_sessions_job_id",
        "query_sessions",
        "jobs",
        ["job_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_query_sessions_job_id", "query_sessions", ["job_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_query_sessions_job_id", table_name="query_sessions")
    op.drop_constraint("fk_query_sessions_job_id", "query_sessions", type_="foreignkey")
    op.drop_column("query_sessions", "job_id")
