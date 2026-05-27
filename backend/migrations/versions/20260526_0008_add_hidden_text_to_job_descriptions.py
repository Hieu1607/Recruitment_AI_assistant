"""Add recruiter-only hidden text to job descriptions.

Revision ID: 20260526_0008
Revises: 20260525_0007
Create Date: 2026-05-26 00:08:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260526_0008"
down_revision = "20260525_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "job_descriptions",
        sa.Column("hidden_text", sa.Text(), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("job_descriptions", "hidden_text")
