"""Add rich outreach message fields and outreach templates.

Revision ID: 20260609_0012
Revises: 20260605_0011
Create Date: 2026-06-09 00:00:01
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260609_0012"
down_revision = "20260605_0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    content_source_enum = postgresql.ENUM(
        "ai_draft",
        "template",
        name="content_source_enum",
        create_type=False,
    )

    op.create_table(
        "outreach_templates",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_by_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("content_source", content_source_enum, nullable=False),
        sa.Column("subject_template", sa.String(length=255), nullable=False),
        sa.Column("body_text_template", sa.Text(), nullable=False),
        sa.Column("body_html_template", sa.Text(), nullable=False),
        sa.Column("editor_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("variables_used", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], name="fk_outreach_templates_job_id", ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id", name="pk_outreach_templates"),
    )
    op.create_index("ix_outreach_templates_created_by_user_id", "outreach_templates", ["created_by_user_id"], unique=False)
    op.create_index("ix_outreach_templates_job_id", "outreach_templates", ["job_id"], unique=False)

    op.add_column("outreach_messages", sa.Column("body_text", sa.Text(), nullable=True))
    op.add_column("outreach_messages", sa.Column("body_html", sa.Text(), nullable=True))
    op.add_column("outreach_messages", sa.Column("template_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("outreach_messages", sa.Column("render_variables", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.create_index("ix_outreach_messages_template_id", "outreach_messages", ["template_id"], unique=False)
    op.create_foreign_key(
        "fk_outreach_messages_template_id",
        "outreach_messages",
        "outreach_templates",
        ["template_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.execute("UPDATE outreach_messages SET body_text = body, body_html = '<p>' || replace(body, E'\\n', '<br>') || '</p>'")
    op.alter_column("outreach_messages", "body_text", nullable=False)
    op.alter_column("outreach_messages", "body_html", nullable=False)
    op.drop_column("outreach_messages", "body")


def downgrade() -> None:
    op.add_column("outreach_messages", sa.Column("body", sa.Text(), nullable=True))
    op.execute("UPDATE outreach_messages SET body = coalesce(body_text, '')")
    op.alter_column("outreach_messages", "body", nullable=False)
    op.drop_constraint("fk_outreach_messages_template_id", "outreach_messages", type_="foreignkey")
    op.drop_index("ix_outreach_messages_template_id", table_name="outreach_messages")
    op.drop_column("outreach_messages", "render_variables")
    op.drop_column("outreach_messages", "template_id")
    op.drop_column("outreach_messages", "body_html")
    op.drop_column("outreach_messages", "body_text")

    op.drop_index("ix_outreach_templates_job_id", table_name="outreach_templates")
    op.drop_index("ix_outreach_templates_created_by_user_id", table_name="outreach_templates")
    op.drop_table("outreach_templates")
