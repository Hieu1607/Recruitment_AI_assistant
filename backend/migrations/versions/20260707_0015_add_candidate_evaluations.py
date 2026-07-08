"""Add candidate evaluations and job scoring preferences.

Revision ID: 20260707_0015
Revises: 20260622_0014
Create Date: 2026-07-07 19:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260707_0015"
down_revision = "20260622_0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    evaluation_status = postgresql.ENUM(
        "pending",
        "running",
        "completed",
        "failed",
        "outdated",
        name="candidate_evaluation_status_enum",
        create_type=False,
    )
    evaluation_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "candidate_evaluations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_description_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("candidate_profile_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("scoring_signature", sa.String(length=128), nullable=False),
        sa.Column("rubric_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("raw_component_scores", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("rationale_summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("status", evaluation_status, nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("source_match_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("scored_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["candidate_profile_id"], ["candidate_profiles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["job_description_id"], ["job_descriptions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_match_run_id"], ["match_runs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "job_description_id",
            "candidate_profile_id",
            "scoring_signature",
            name="uq_candidate_evaluations_jd_candidate_signature",
        ),
    )
    op.create_index("ix_candidate_evaluations_job_id", "candidate_evaluations", ["job_id"], unique=False)
    op.create_index(
        "ix_candidate_evaluations_job_description_id",
        "candidate_evaluations",
        ["job_description_id"],
        unique=False,
    )
    op.create_index(
        "ix_candidate_evaluations_candidate_profile_id",
        "candidate_evaluations",
        ["candidate_profile_id"],
        unique=False,
    )
    op.create_index(
        "ix_candidate_evaluations_scoring_signature",
        "candidate_evaluations",
        ["scoring_signature"],
        unique=False,
    )

    op.create_table(
        "job_scoring_preferences",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("section_weights", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("score_threshold", sa.Numeric(5, 2), nullable=False, server_default="50.00"),
        sa.Column("updated_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["updated_by_user_id"], ["user_accounts.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("job_id"),
    )


def downgrade() -> None:
    op.drop_table("job_scoring_preferences")
    op.drop_index("ix_candidate_evaluations_scoring_signature", table_name="candidate_evaluations")
    op.drop_index("ix_candidate_evaluations_candidate_profile_id", table_name="candidate_evaluations")
    op.drop_index("ix_candidate_evaluations_job_description_id", table_name="candidate_evaluations")
    op.drop_index("ix_candidate_evaluations_job_id", table_name="candidate_evaluations")
    op.drop_table("candidate_evaluations")
    postgresql.ENUM(name="candidate_evaluation_status_enum").drop(op.get_bind(), checkfirst=True)
