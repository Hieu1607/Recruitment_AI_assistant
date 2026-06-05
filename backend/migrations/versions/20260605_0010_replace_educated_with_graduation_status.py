"""Replace educated with graduation_status on candidate profiles.

Revision ID: 20260605_0010
Revises: 20260528_0009
Create Date: 2026-06-05 00:00:00
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260605_0010"
down_revision = "20260528_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "candidate_profiles",
        sa.Column(
            "graduation_status",
            sa.String(length=50),
            nullable=False,
            server_default="unknown",
        ),
    )
    op.drop_column("candidate_profiles", "educated")


def downgrade() -> None:
    op.add_column(
        "candidate_profiles",
        sa.Column(
            "educated",
            sa.Boolean(),
            nullable=True,
            server_default=sa.text("false"),
        ),
    )
    op.alter_column("candidate_profiles", "educated", nullable=False, server_default=sa.text("false"))
    op.drop_column("candidate_profiles", "graduation_status")
