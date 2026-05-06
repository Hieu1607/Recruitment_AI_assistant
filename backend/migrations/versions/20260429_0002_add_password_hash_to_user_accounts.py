"""Add password_hash to user_accounts.

Revision ID: 20260429_0002
Revises: 20260324_0001
Create Date: 2026-04-29 00:00:00
"""

from alembic import op
import sqlalchemy as sa

revision = "20260429_0002"
down_revision = "20260324_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "user_accounts",
        sa.Column("password_hash", sa.String(255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("user_accounts", "password_hash")
