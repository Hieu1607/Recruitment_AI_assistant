"""Initial recruitment schema.

Revision ID: 20260324_0001
Revises:
Create Date: 2026-03-24 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "20260324_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    upload_status_enum = postgresql.ENUM(
        "uploaded", "processing", "processed", "failed", name="upload_status_enum", create_type=False
    )
    profile_status_enum = postgresql.ENUM(
        "draft", "reviewed", "approved", "archived", name="profile_status_enum", create_type=False
    )
    match_run_status_enum = postgresql.ENUM(
        "running", "completed", "failed", name="match_run_status_enum", create_type=False
    )
    content_source_enum = postgresql.ENUM(
        "ai_draft", "template", name="content_source_enum", create_type=False
    )
    sent_status_enum = postgresql.ENUM(
        "not_sent", "sent", "failed", name="sent_status_enum", create_type=False
    )
    user_status_enum = postgresql.ENUM(
        "active", "suspended", name="user_status_enum", create_type=False
    )
    role_name_enum = postgresql.ENUM(
        "admin", "recruiter", "viewer", name="role_name_enum", create_type=False
    )

    bind = op.get_bind()
    upload_status_enum.create(bind, checkfirst=True)
    profile_status_enum.create(bind, checkfirst=True)
    match_run_status_enum.create(bind, checkfirst=True)
    content_source_enum.create(bind, checkfirst=True)
    sent_status_enum.create(bind, checkfirst=True)
    user_status_enum.create(bind, checkfirst=True)
    role_name_enum.create(bind, checkfirst=True)

    op.create_table(
        "resume_documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("original_file_name", sa.String(length=255), nullable=False),
        sa.Column("storage_uri", sa.String(length=1024), nullable=False),
        sa.Column("upload_status", upload_status_enum, nullable=False, server_default="uploaded"),
        sa.Column("duplicate_group_key", sa.String(length=255), nullable=True),
        sa.Column("uploaded_by_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retention_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_resume_documents"),
    )

    op.create_table(
        "user_accounts",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("status", user_status_enum, nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_user_accounts"),
        sa.UniqueConstraint("email", name="uq_user_accounts_email"),
    )
    op.create_index("ix_user_accounts_email", "user_accounts", ["email"], unique=False)

    op.create_table(
        "candidate_profiles",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("resume_document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=False),
        sa.Column("phone", sa.String(length=50), nullable=True),
        sa.Column("email", sa.String(length=320), nullable=True),
        sa.Column("location_normalized", sa.String(length=255), nullable=True),
        sa.Column("contact", sa.String(length=255), nullable=True),
        sa.Column("current_job_title", sa.String(length=255), nullable=True),
        sa.Column("educated", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("ever_studied_abroad", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("major", sa.String(length=255), nullable=True),
        sa.Column("cpa", sa.String(length=100), nullable=True),
        sa.Column("education_text", sa.Text(), nullable=True),
        sa.Column("experience_text", sa.Text(), nullable=True),
        sa.Column("experience_years", sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column("skills_text", sa.Text(), nullable=True),
        sa.Column("languages_text", sa.Text(), nullable=True),
        sa.Column("projects_text", sa.Text(), nullable=True),
        sa.Column("summary_text", sa.Text(), nullable=True),
        sa.Column("achievements_text", sa.Text(), nullable=True),
        sa.Column("publications_text", sa.Text(), nullable=True),
        sa.Column("certifications_text", sa.Text(), nullable=True),
        sa.Column("references_text", sa.Text(), nullable=True),
        sa.Column("other_text", sa.Text(), nullable=True),
        sa.Column("profile_status", profile_status_enum, nullable=False, server_default="draft"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["resume_document_id"], ["resume_documents.id"], name="fk_cp_resume_doc", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_candidate_profiles"),
        sa.UniqueConstraint("resume_document_id", name="uq_candidate_profiles_resume_document_id"),
    )
    op.create_index("ix_candidate_profiles_email", "candidate_profiles", ["email"], unique=False)

    op.create_table(
        "extraction_traces",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("resume_document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stage", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["resume_document_id"], ["resume_documents.id"], name="fk_et_resume_doc", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_extraction_traces"),
    )
    op.create_index("ix_extraction_traces_resume_document_id", "extraction_traces", ["resume_document_id"], unique=False)

    op.create_table(
        "job_descriptions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("jd_text", sa.Text(), nullable=False),
        sa.Column("created_by_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.PrimaryKeyConstraint("id", name="pk_job_descriptions"),
    )

    op.create_table(
        "match_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_description_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("score_threshold", sa.Numeric(precision=5, scale=2), nullable=False),
        sa.Column("initiated_by_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_status", match_run_status_enum, nullable=False, server_default="running"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("score_threshold >= 0 AND score_threshold <= 100", name="ck_match_runs_score_threshold_range"),
        sa.ForeignKeyConstraint(["job_description_id"], ["job_descriptions.id"], name="fk_mr_job_desc", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_match_runs"),
    )
    op.create_index("ix_match_runs_job_description_id", "match_runs", ["job_description_id"], unique=False)

    op.create_table(
        "query_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_title", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_query_sessions"),
    )
    op.create_index("ix_query_sessions_user_id", "query_sessions", ["user_id"], unique=False)

    op.create_table(
        "role_assignments",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role_name", role_name_enum, nullable=False),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["user_id"], ["user_accounts.id"], name="fk_ra_user", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_role_assignments"),
        sa.UniqueConstraint("user_id", "role_name", name="uq_role_assignment_user_role"),
    )
    op.create_index("ix_role_assignments_user_id", "role_assignments", ["user_id"], unique=False)

    op.create_table(
        "interview_question_sets",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("candidate_profile_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_description_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("generated_by_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("question_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["candidate_profile_id"], ["candidate_profiles.id"], name="fk_iqs_candidate", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["job_description_id"], ["job_descriptions.id"], name="fk_iqs_job_desc", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_interview_question_sets"),
    )
    op.create_index("ix_interview_question_sets_candidate_profile_id", "interview_question_sets", ["candidate_profile_id"], unique=False)
    op.create_index("ix_interview_question_sets_job_description_id", "interview_question_sets", ["job_description_id"], unique=False)

    op.create_table(
        "match_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("match_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("candidate_profile_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("score_list_index", sa.Integer(), nullable=False),
        sa.Column("total_score", sa.Numeric(precision=5, scale=2), nullable=False),
        sa.Column("passed_threshold", sa.Boolean(), nullable=False),
        sa.Column("rationale_summary", sa.Text(), nullable=False),
        sa.Column("component_scores", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.CheckConstraint("total_score >= 0 AND total_score <= 100", name="ck_match_results_total_score_range"),
        sa.ForeignKeyConstraint(["candidate_profile_id"], ["candidate_profiles.id"], name="fk_mres_candidate", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["match_run_id"], ["match_runs.id"], name="fk_mres_match_run", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_match_results"),
        sa.UniqueConstraint("match_run_id", "candidate_profile_id", name="uq_match_results_run_candidate"),
    )
    op.create_index("ix_match_results_candidate_profile_id", "match_results", ["candidate_profile_id"], unique=False)
    op.create_index("ix_match_results_match_run_id", "match_results", ["match_run_id"], unique=False)

    op.create_table(
        "outreach_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("candidate_profile_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_by_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("content_source", content_source_enum, nullable=False),
        sa.Column("subject", sa.String(length=255), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("sent_status", sent_status_enum, nullable=False, server_default="not_sent"),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["candidate_profile_id"], ["candidate_profiles.id"], name="fk_om_candidate", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_outreach_messages"),
    )
    op.create_index("ix_outreach_messages_candidate_profile_id", "outreach_messages", ["candidate_profile_id"], unique=False)

    op.create_table(
        "query_turns",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("query_session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_question", sa.Text(), nullable=False),
        sa.Column("answer_text", sa.Text(), nullable=False),
        sa.Column("matched_candidate_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("matched_count", sa.Integer(), nullable=True),
        sa.Column("tool_trace_masked", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.CheckConstraint("matched_count IS NULL OR matched_count >= 0", name="ck_query_turns_matched_count_non_negative"),
        sa.ForeignKeyConstraint(["query_session_id"], ["query_sessions.id"], name="fk_qt_session", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_query_turns"),
    )
    op.create_index("ix_query_turns_query_session_id", "query_turns", ["query_session_id"], unique=False)

    op.create_table(
        "shortlist_collections",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("created_by_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_query_turn_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["source_query_turn_id"], ["query_turns.id"], name="fk_sc_source_turn", ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id", name="pk_shortlist_collections"),
        sa.UniqueConstraint("created_by_user_id", "name", name="uq_shortlist_creator_name"),
    )
    op.create_index("ix_shortlist_collections_created_by_user_id", "shortlist_collections", ["created_by_user_id"], unique=False)

    op.create_table(
        "shortlist_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("shortlist_collection_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("candidate_profile_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("added_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["candidate_profile_id"], ["candidate_profiles.id"], name="fk_si_candidate", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["shortlist_collection_id"], ["shortlist_collections.id"], name="fk_si_collection", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_shortlist_items"),
        sa.UniqueConstraint("shortlist_collection_id", "candidate_profile_id", name="uq_shortlist_item_unique"),
    )
    op.create_index("ix_shortlist_items_candidate_profile_id", "shortlist_items", ["candidate_profile_id"], unique=False)
    op.create_index("ix_shortlist_items_shortlist_collection_id", "shortlist_items", ["shortlist_collection_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_shortlist_items_shortlist_collection_id", table_name="shortlist_items")
    op.drop_index("ix_shortlist_items_candidate_profile_id", table_name="shortlist_items")
    op.drop_table("shortlist_items")

    op.drop_index("ix_shortlist_collections_created_by_user_id", table_name="shortlist_collections")
    op.drop_table("shortlist_collections")

    op.drop_index("ix_query_turns_query_session_id", table_name="query_turns")
    op.drop_table("query_turns")

    op.drop_index("ix_outreach_messages_candidate_profile_id", table_name="outreach_messages")
    op.drop_table("outreach_messages")

    op.drop_index("ix_match_results_match_run_id", table_name="match_results")
    op.drop_index("ix_match_results_candidate_profile_id", table_name="match_results")
    op.drop_table("match_results")

    op.drop_index("ix_interview_question_sets_job_description_id", table_name="interview_question_sets")
    op.drop_index("ix_interview_question_sets_candidate_profile_id", table_name="interview_question_sets")
    op.drop_table("interview_question_sets")

    op.drop_index("ix_role_assignments_user_id", table_name="role_assignments")
    op.drop_table("role_assignments")

    op.drop_index("ix_query_sessions_user_id", table_name="query_sessions")
    op.drop_table("query_sessions")

    op.drop_index("ix_match_runs_job_description_id", table_name="match_runs")
    op.drop_table("match_runs")

    op.drop_table("job_descriptions")

    op.drop_index("ix_extraction_traces_resume_document_id", table_name="extraction_traces")
    op.drop_table("extraction_traces")

    op.drop_index("ix_candidate_profiles_email", table_name="candidate_profiles")
    op.drop_table("candidate_profiles")

    op.drop_index("ix_user_accounts_email", table_name="user_accounts")
    op.drop_table("user_accounts")

    op.drop_table("resume_documents")

    bind = op.get_bind()
    sa.Enum(name="role_name_enum").drop(bind, checkfirst=True)
    sa.Enum(name="user_status_enum").drop(bind, checkfirst=True)
    sa.Enum(name="sent_status_enum").drop(bind, checkfirst=True)
    sa.Enum(name="content_source_enum").drop(bind, checkfirst=True)
    sa.Enum(name="match_run_status_enum").drop(bind, checkfirst=True)
    sa.Enum(name="profile_status_enum").drop(bind, checkfirst=True)
    sa.Enum(name="upload_status_enum").drop(bind, checkfirst=True)
