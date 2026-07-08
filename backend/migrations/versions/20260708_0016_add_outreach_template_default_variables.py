"""Add default_variables to outreach_templates.

Revision ID: 20260708_0016
Revises: 20260707_0015
Create Date: 2026-07-08 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260708_0016"
down_revision = "20260707_0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "outreach_templates",
        sa.Column("default_variables", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("outreach_templates", "default_variables")
