"""Add user notifications.

Revision ID: 20260622_0014
Revises: 20260612_0013
Create Date: 2026-06-22 17:10:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260622_0014"
down_revision = "20260612_0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_notifications",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("notification_type", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("body", sa.Text(), server_default="", nullable=False),
        sa.Column("target_url", sa.String(length=512), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("read_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["user_accounts.id"], name="fk_user_notifications_user_accounts", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_user_notifications"),
    )
    op.create_index("ix_user_notifications_user_id", "user_notifications", ["user_id"], unique=False)
    op.create_index("ix_user_notifications_notification_type", "user_notifications", ["notification_type"], unique=False)
    op.create_index("ix_user_notifications_created_at", "user_notifications", ["created_at"], unique=False)
    op.create_index("ix_user_notifications_read_at", "user_notifications", ["read_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_user_notifications_read_at", table_name="user_notifications")
    op.drop_index("ix_user_notifications_created_at", table_name="user_notifications")
    op.drop_index("ix_user_notifications_notification_type", table_name="user_notifications")
    op.drop_index("ix_user_notifications_user_id", table_name="user_notifications")
    op.drop_table("user_notifications")
