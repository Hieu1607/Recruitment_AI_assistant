"""Add structured profile JSON to candidate profiles.

Revision ID: 20260525_0007
Revises: 20260522_0006
Create Date: 2026-05-25 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "20260525_0007"
down_revision = "20260522_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "candidate_profiles",
        sa.Column(
            "structured_profile",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("candidate_profiles", "structured_profile")
