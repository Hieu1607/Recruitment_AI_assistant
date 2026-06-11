"""Add Google mail token storage fields.

Revision ID: 20260605_0011
Revises: 20260605_0010
Create Date: 2026-06-05 00:00:01
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260605_0011"
down_revision = "20260605_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("oauth_identities", sa.Column("access_token_encrypted", sa.Text(), nullable=True))
    op.add_column("oauth_identities", sa.Column("refresh_token_encrypted", sa.Text(), nullable=True))
    op.add_column("oauth_identities", sa.Column("token_expires_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("oauth_identities", sa.Column("scope", sa.Text(), nullable=True))
    op.add_column(
        "oauth_identities",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("oauth_identities", "updated_at")
    op.drop_column("oauth_identities", "scope")
    op.drop_column("oauth_identities", "token_expires_at")
    op.drop_column("oauth_identities", "refresh_token_encrypted")
    op.drop_column("oauth_identities", "access_token_encrypted")
