"""Expand candidate profile cpa field to text.

Revision ID: 20260612_0013
Revises: 20260609_0012
Create Date: 2026-06-12 00:00:00
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260612_0013"
down_revision = "20260609_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "candidate_profiles",
        "cpa",
        existing_type=sa.String(length=100),
        type_=sa.Text(),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "candidate_profiles",
        "cpa",
        existing_type=sa.Text(),
        type_=sa.String(length=100),
        existing_nullable=True,
    )
