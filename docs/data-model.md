# Data Model: Recruitment AI Assistant Website

## 1. ResumeDocument
- Purpose: Represents uploaded CV files and extraction lifecycle.
- Fields:
  - id (UUID)
  - original_file_name (string, required)
  - storage_uri (string, required)
  - upload_status (enum: uploaded, processing, processed, failed)
  - duplicate_group_key (string, nullable)
  - uploaded_by_user_id (UUID, required)
  - uploaded_at (timestamp, required)
  - processed_at (timestamp, nullable)
  - retention_expires_at (timestamp, required)
- Relationships:
  - 1-to-1 with CandidateProfile revisions.
  - 1-to-many with ExtractionTrace.
- Validation Rules:
  - Accept only PDF content type.
  - retention_expires_at = uploaded_at + 12 months.

## 2. CandidateProfile
- Purpose: Structured candidate record normalized from resume content.
- Fields:
  - id (UUID)
  - resume_document_id (UUID, required)
  - full_name (string, required)
  - phone (string, nullable)
  - email (string, nullable, format validation)
  - location_normalized (string, nullable)
  - contact (string, nullable, e.g Linkedin, Github)
  - current_job_title (string, nullable)
  - graduation_status (string enum: graduated, final_year, studying, unknown; required)
  - ever_studied_abroad (boolean, required)
  - major (string, nullable)
  - cpa (string, nullable)
  - education_text (text, nullable)
  - experience_text (text, nullable)
  - experience_years (numeric, nullable)
  - skills_text (text, nullable)
  - languages_text (text, nullable)
  - projects_text (text, nullable)
  - summary_text (text, nullable)
  - achievements_text (text, nullable)
  - publications_text (text, nullable)
  - certifications_text (text, nullable)
  - references_text (text, nullable)
  - other_text (text, nullable)
  - profile_status (enum: draft, reviewed, approved, archived)
  - created_at (timestamp)
  - updated_at (timestamp)
- Relationships:
  - 1-to-1 with ResumeDocument.
  - 1-to-many with MatchResult, QueryEvidence, OutreachMessage, InterviewQuestionSet.
- Validation Rules:
  - Missing optional text fields stored as null.
  - Missing evidence booleans default false.
  - location_normalized must match allowed format (province/city, province+country, or country only).

## 4. JobDescription
- Purpose: Stores JD inputs used for scoring and interview generation.
- Fields:
  - id (UUID)
  - title (string, nullable)
  - jd_text (text, required)
  - created_by_user_id (UUID, required)
  - created_at (timestamp)
  - is_active (boolean)
- Relationships:
  - 1-to-many with MatchRun and InterviewQuestionSet.

## 5. MatchRun
- Purpose: Represents a scoring execution for candidate set against one JD.
- Fields:
  - id (UUID)
  - job_description_id (UUID, required)
  - score_threshold (UUID, required, 0..100)
  - initiated_by_user_id (UUID, required)
  - run_status (enum: running, completed, failed)
  - created_at (timestamp)
  - completed_at (timestamp, nullable)
- Validation Rules:
  - one MatchRun evaluates one JD against many candidates in a single batch call.

## 6. MatchResult
- Purpose: Candidate-specific output from a MatchRun.
- Fields:
  - id (UUID)
  - match_run_id (UUID, required)
  - candidate_profile_id (UUID, required)
  - score_list_index (integer, required)
  - total_score (numeric, required, 0..100)
  - passed_threshold (boolean, required)
  - rationale_summary (text, required)
  - component_scores (json array, required) with items:
    - criterion_key (string, required; e.g., skills, education, experience)
    - weight (numeric, required, 0..1)
    - score (numeric, required, 0..100)
    - weighted_score (numeric, required, 0..100)
    - evidence_summary (text, nullable)
  - created_at (timestamp)
- Validation Rules:
  - total_score normalized to 0..100 and computed from component_scores according to configured weights.
  - passed_threshold = total_score >= score_threshold at run time.

## 7. QuerySession and QueryTurn
- Purpose: Maintains recruiter chat interactions and filtering outcomes.
- QuerySession Fields:
  - id (UUID)
  - user_id (UUID)
  - session_title (string, nullable)
  - created_at (timestamp)
  - updated_at (timestamp)
- QueryTurn Fields:
  - id (UUID)
  - query_session_id (UUID, required)
  - user_question (text, required)
  - answer_text (text, required)
  - matched_candidate_ids (json, nullable)
  - matched_count (integer, nullable)
  - tool_trace_masked (json, nullable)
  - created_at (timestamp)

## 8. ShortlistCollection and ShortlistItem
- Purpose: Persist recruiter-saved filtered candidate sets.
- ShortlistCollection Fields:
  - id (UUID)
  - name (string, required)
  - created_by_user_id (UUID, required)
  - source_query_turn_id (UUID, nullable)
  - created_at (timestamp)
- ShortlistItem Fields:
  - id (UUID)
  - shortlist_collection_id (UUID, required)
  - candidate_profile_id (UUID, required)
  - added_at (timestamp)
- Validation Rules:
  - Collection name unique per creator within active period.

## 9. OutreachMessage
- Purpose: Stores AI/template drafts and human-approved sends.
- Fields:
  - id (UUID)
  - candidate_profile_id (UUID, required)
  - created_by_user_id (UUID, required)
  - content_source (enum: ai_draft, template)
  - subject (string, required)
  - body (text, required)
  - sent_status (enum: not_sent, sent, failed)
  - sent_at (timestamp, nullable)
  - created_at (timestamp)

## 10. InterviewQuestionSet
- Purpose: Generated interview questions for candidate + JD pair.
- Fields:
  - id (UUID)
  - candidate_profile_id (UUID, required)
  - job_description_id (UUID, required)
  - generated_by_user_id (UUID, required)
  - question_payload (json, required)
  - created_at (timestamp)

## 11. UserAccount and RoleAssignment
- Purpose: RBAC support for Admin/Recruiter/Viewer permissions.
- UserAccount Fields:
  - id (UUID)
  - email (string, required)
  - display_name (string, required)
  - status (enum: active, suspended)
  - created_at (timestamp)
- RoleAssignment Fields:
  - id (UUID)
  - user_id (UUID, required)
  - role_name (enum: admin, recruiter, viewer)
  - assigned_at (timestamp)

