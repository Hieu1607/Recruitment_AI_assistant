/** Generated from docs/BACKEND.md. Update both when the API changes. */

// ---------------------------------------------------------------------------
// Enums (string union types)
// ---------------------------------------------------------------------------

export type UploadStatus = "uploaded" | "processing" | "processed" | "failed";

export type ProfileStatus = "draft" | "reviewed" | "approved" | "archived";

export type MatchRunStatus = "running" | "completed" | "failed";

export type ContentSource = "ai_draft" | "template";

export type SentStatus = "not_sent" | "sent" | "failed";

export type UserStatus = "active" | "suspended";

export type RoleName = "admin" | "recruiter" | "viewer";

// ---------------------------------------------------------------------------
// Resume Upload & Management
// ---------------------------------------------------------------------------

export interface ResumeResponse {
  id: string;                              // UUID
  job_id?: string;                         // UUID
  original_file_name: string;
  candidate_profile_id?: string | null;    // UUID
  candidate_display_name?: string | null;
  storage_uri: string;                     // server-side file path
  upload_status: UploadStatus;
  duplicate_group_key: string | null;
  uploaded_by_user_id: string;             // UUID
  uploader_display_name?: string | null;
  uploaded_at: string | null;              // ISO 8601 datetime
  processed_at: string | null;             // ISO 8601 datetime
  retention_expires_at: string | null;     // ISO 8601 datetime
}

export interface ResumeBatchParseItem {
  file_name: string;
  resume_document_id: string;             // UUID
  candidate_profile_id?: string | null;   // UUID — only present for synchronous/legacy flows
  task_id?: string | null;                // Celery task id for async queueing
  status: "queued" | UploadStatus;
}

export interface ResumeBatchParseResponse {
  total_files: number;
  queued_files: number;
  items: ResumeBatchParseItem[];
}

export interface ResumeListResponse {
  total: number;
  items: ResumeResponse[];
}

export interface StructuredLink {
  url: string;
  label: string | null;
}

export interface StructuredEntry {
  title: string | null;
  subtitle: string | null;
  role: string | null;
  location: string | null;
  dateRange: string | null;
  description: string | null;
  bullets: string[];
  links: StructuredLink[];
  metadata: string[];
}

export interface StructuredSection {
  entries: StructuredEntry[];
  rawText: string | null;
}

export interface StructuredSummary {
  text: string | null;
  links: StructuredLink[];
}

export interface StructuredProfile {
  summary?: StructuredSummary | null;
  experience?: StructuredSection | null;
  education?: StructuredSection | null;
  projects?: StructuredSection | null;
  skills?: StructuredSection | null;
  languages?: StructuredSection | null;
  achievements?: StructuredSection | null;
  publications?: StructuredSection | null;
  certifications?: StructuredSection | null;
  references?: StructuredSection | null;
  other?: StructuredSection | null;
}

export interface CandidateProfileResponse {
  id: string;
  resume_document_id: string;
  extraction_mode?: string | null;
  full_name: string;
  submitted_full_name: string | null;
  phone: string | null;
  email: string | null;
  submitted_email: string | null;
  location_normalized: string | null;
  contact: string | null;
  current_job_title: string | null;
  graduation_status: string;
  ever_studied_abroad: boolean;
  major: string | null;
  cpa: string | null;
  summary_text: string | null;
  skills_text: string | null;
  experience_text: string | null;
  experience_years: number | null;
  education_text: string | null;
  languages_text: string | null;
  projects_text: string | null;
  achievements_text: string | null;
  publications_text: string | null;
  certifications_text: string | null;
  references_text: string | null;
  other_text: string | null;
  structured_profile: StructuredProfile | null;
}

// ---------------------------------------------------------------------------
// Job Descriptions
// ---------------------------------------------------------------------------

export interface JobDescriptionResponse {
  id: string;                              // UUID
  job_id?: string;                         // UUID of the owning job/workspace
  title: string | null;
  jd_text: string;
  hidden_text: string;
  created_by_user_id: string;             // UUID
  created_at: string;                     // ISO 8601 datetime
  is_active: boolean;
}

export interface JobDescriptionListResponse {
  total: number;
  items: JobDescriptionResponse[];
}

export interface JobDescriptionCreateRequest {
  title?: string;
  jd_text: string;
  hidden_text?: string;
  created_by_user_id: string;             // UUID
}

export interface JobDescriptionUpdateRequest {
  title?: string;
  jd_text?: string;
  hidden_text?: string;
  is_active?: boolean;
}

// ---------------------------------------------------------------------------
// Candidate Scoring
// ---------------------------------------------------------------------------

export interface SectionWeights {
  skills?: number;
  experience?: number;
  education?: number;
  projects?: number;
  summary?: number;
  languages?: number;
  achievements?: number;
  certifications?: number;
  publications?: number;
  other?: number;
}

export interface ScoreRequest {
  job_description_id: string;             // UUID
  initiated_by_user_id: string;           // UUID
  score_threshold?: number;              // 0 - 100, default 50
  candidate_profile_ids?: string[];      // UUIDs — omit to score all
  section_weights?: SectionWeights;
  batch_size?: number;                   // 1 - 50, default 10
}

/**
 * NOTE: The `scores` array uses camelCase keys because the LLM returns them
 * as-is (BACKEND.md note 7). All other fields in this response are snake_case.
 */
export interface ComponentScore {
  criterionKey: string;                  // e.g. "skills"
  criterionType?: string | null;
  evaluationMode?: string | null;
  requirementText?: string | null;
  weight: number;
  score: number;
  weightedScore: number;
  evidenceSummary: string;
}

export interface CandidateScore {
  candidateId: string;                   // UUID — camelCase per BACKEND.md note 7
  candidateName?: string | null;
  resumeFileName?: string | null;
  candidateDisplayName?: string | null;
  totalScore: number;                    // 0 - 100
  passedThreshold: boolean;
  rationale: string;
  componentScores: ComponentScore[];
}

export interface ScoreResponse {
  match_run_id: string;                  // UUID
  job_description_id: string;           // UUID
  total_candidates: number;
  total_passed_candidates: number;
  batches: number;
  scores: CandidateScore[];
}

// ---------------------------------------------------------------------------
// Chat / Recruiter Chatbot
// ---------------------------------------------------------------------------

export interface ChatRequest {
  message: string;
  session_id?: string;
  candidate_limit?: number;             // 1 - 2000, default 500
}

export interface JobResponse {
  id: string;
  owner_user_id: string;
  title: string;
  status: string;
  candidate_message: string | null;
  public_apply_enabled: boolean;
  public_apply_url: string;
  created_at: string;
  updated_at: string;
  archived_at: string | null;
}

export interface JobApplicationLinkResponse {
  public_apply_enabled: boolean;
  public_apply_url: string;
  candidate_message: string | null;
}

export interface JobListResponse {
  items: JobResponse[];
  total: number;
}

export interface PublicJobResponse {
  job_title: string;
  candidate_message: string | null;
  public_apply_enabled: boolean;
}

export interface PublicResumeUploadResponse {
  submitted: boolean;
  candidate_profile_id: string | null;
}

export interface ChatResponse {
  session_id: string;
  answer: string;
  candidates_in_scope: number;
  session?: ChatSessionResponse | null;
  turn?: ChatTurnResponse | null;
}

export interface ChatSessionResponse {
  id: string;
  user_id: string;
  job_id: string;
  session_title: string | null;
  turn_count: number;
  created_at: string;
  updated_at: string;
}

export interface ChatSessionListResponse {
  items: ChatSessionResponse[];
  total: number;
}

export interface ChatTurnResponse {
  id: string;
  query_session_id: string;
  user_question: string;
  answer_text: string;
  matched_candidate_ids: string[] | null;
  matched_count: number | null;
  tool_trace_masked: Record<string, unknown> | null;
  created_at: string;
}

export interface JobSetupStatusResponse {
  job_id: string;
  resume_count: number;
  processed_candidate_count: number;
  has_uploaded_resumes: boolean;
  has_processed_candidates: boolean;
  has_active_job_description: boolean;
  has_completed_score_run: boolean;
  has_chat_turn: boolean;
  completed_score_run_count: number;
  chat_session_count: number;
  chat_turn_count: number;
  latest_job_description_id: string | null;
  latest_score_run_id: string | null;
  latest_score_run_at: string | null;
  latest_chat_session_id: string | null;
  latest_chat_turn_at: string | null;
}

export interface ChatMessage {
  role: "human" | "ai";
  content: string;
}

export interface ChatHistoryResponse {
  session_id: string;
  messages: ChatMessage[];
}

export interface ChatDeleteResponse {
  session_id: string;
  deleted: boolean;
}

// ---------------------------------------------------------------------------
// Shortlist — Query Sessions
// ---------------------------------------------------------------------------

export interface SessionResponse {
  id: string;                            // UUID
  user_id: string;                       // UUID
  session_title: string | null;
  turn_count: number;
  created_at: string;                    // ISO 8601 datetime
  updated_at: string;                    // ISO 8601 datetime
}

export interface SessionListResponse {
  items: SessionResponse[];
  total?: number;
}

export interface SessionCreateRequest {
  user_id: string;                       // UUID
  session_title?: string;
}

export interface SessionUpdateRequest {
  session_title: string;
}

// ---------------------------------------------------------------------------
// Shortlist — Query Turns
// ---------------------------------------------------------------------------

export interface TurnResponse {
  id: string;                            // UUID
  query_session_id: string;             // UUID
  user_question: string;
  answer_text: string;
  matched_candidate_ids: string[] | null; // UUIDs
  matched_count: number | null;
  tool_trace_masked: Record<string, unknown> | null;
  created_at: string;                    // ISO 8601 datetime
}

export interface TurnListResponse {
  items: TurnResponse[];
  total?: number;
}

export interface TurnCreateRequest {
  user_question: string;
  answer_text: string;
  matched_candidate_ids?: string[];
  matched_count?: number;
  tool_trace_masked?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Shortlist — Collections
// ---------------------------------------------------------------------------

export interface CollectionResponse {
  id: string;                            // UUID
  name: string;
  created_by_user_id: string;           // UUID
  source_query_turn_id: string | null;  // UUID
  item_count: number;
  created_at: string;                    // ISO 8601 datetime
}

export interface CollectionListResponse {
  items: CollectionResponse[];
  total?: number;
}

export interface CollectionCreateRequest {
  created_by_user_id: string;           // UUID
  name: string;
  source_query_turn_id?: string;        // UUID
}

export interface CollectionUpdateRequest {
  name: string;
}

// ---------------------------------------------------------------------------
// Shortlist — Items
// ---------------------------------------------------------------------------

export interface ShortlistItemResponse {
  id: string;                            // UUID
  shortlist_collection_id: string;      // UUID
  candidate_profile_id: string;         // UUID
  added_at: string;                     // ISO 8601 datetime
}

export interface ShortlistItemListResponse {
  items: ShortlistItemResponse[];
  total?: number;
}

export interface ShortlistItemCreateRequest {
  candidate_profile_id: string;         // UUID
}

// ---------------------------------------------------------------------------
// Shortlist — Dispatch
// ---------------------------------------------------------------------------

export interface DispatchCollectionResponse {
  id: string;
  name: string;
  item_count: number;
}

export interface DispatchJobResponse {
  id: string;
  title: string;
}

export interface DispatchOutreachStatus {
  latest_message_id: string;
  status: SentStatus;
  created_at: string;
  sent_at: string | null;
}

export interface DispatchInterviewStatus {
  latest_invitation_id: string;
  status: string;
  interview_template_id: string;
  template_name: string | null;
  sent_at: string | null;
  completed_at: string | null;
}

export interface DispatchCandidateResponse {
  candidate_profile_id: string;
  full_name: string;
  email: string | null;
  current_job_title: string | null;
  skills_text: string | null;
  contact_status: "ready" | "missing_email";
  outreach: DispatchOutreachStatus | null;
  interview: DispatchInterviewStatus | null;
  blockers: string[];
}

export interface DispatchCapabilitiesResponse {
  gmail_connected: boolean;
  active_interview_templates_count: number;
}

export interface DispatchSummaryResponse {
  collection: DispatchCollectionResponse;
  job: DispatchJobResponse | null;
  candidates: DispatchCandidateResponse[];
  capabilities: DispatchCapabilitiesResponse;
}

export interface OutreachDraftBatchRequest {
  candidate_profile_ids: string[];
  subject_template: string;
  body_text_template?: string | null;
  body_html_template?: string | null;
  content_source?: ContentSource;
  template_id?: string | null;
  force_update?: boolean;
}

export interface InterviewInvitationBatchRequest {
  candidate_profile_ids: string[];
  job_id: string;
  interview_template_id?: string | null;
  interview_question_set_id?: string | null;
  expires_in_hours?: number | null;
  send_email?: boolean;
}

export interface BatchCandidateResult {
  candidate_profile_id: string;
  full_name: string | null;
  status: string;
  reason: string | null;
  record_id: string | null;
}

export interface BatchActionResponse {
  created_count: number;
  skipped_count: number;
  failed_count: number;
  results: BatchCandidateResult[];
}

// ---------------------------------------------------------------------------
// Outreach Messages
// ---------------------------------------------------------------------------

export interface OutreachResponse {
  id: string;                            // UUID
  candidate_profile_id: string;         // UUID
  candidate_full_name: string | null;
  created_by_user_id: string;           // UUID
  content_source: ContentSource;
  subject: string;
  body_text: string;
  body_html: string;
  template_id: string | null;
  render_variables: Record<string, string> | null;
  sent_status: SentStatus;
  sent_at: string | null;               // ISO 8601 datetime
  created_at: string;                    // ISO 8601 datetime
}

export interface OutreachListResponse {
  total: number;
  items: OutreachResponse[];
}

export interface OutreachCreateRequest {
  candidate_profile_id: string;         // UUID
  created_by_user_id: string;           // UUID
  content_source: ContentSource;
  subject: string;
  body_text: string;
  body_html: string;
  template_id?: string | null;
  render_variables?: Record<string, string> | null;
}

export interface OutreachUpdateRequest {
  subject?: string;
  body_text?: string;
  body_html?: string;
  sent_status?: SentStatus;
}

export interface OutreachTemplateResponse {
  id: string;
  created_by_user_id: string;
  job_id: string | null;
  name: string;
  content_source: ContentSource;
  subject_template: string;
  body_text_template: string;
  body_html_template: string;
  editor_json: Record<string, unknown> | null;
  variables_used: string[];
  created_at: string;
  updated_at: string;
}

export interface OutreachTemplateListResponse {
  total: number;
  items: OutreachTemplateResponse[];
}

export interface OutreachTemplateCreateRequest {
  created_by_user_id: string;
  job_id?: string | null;
  name: string;
  content_source?: ContentSource;
  subject_template: string;
  body_text_template: string;
  body_html_template: string;
  editor_json?: Record<string, unknown> | null;
  variables_used?: string[];
}

export interface OutreachTemplateUpdateRequest {
  name?: string;
  subject_template?: string;
  body_text_template?: string;
  body_html_template?: string;
  editor_json?: Record<string, unknown> | null;
  variables_used?: string[];
}

export interface OutreachAssetUploadResponse {
  storage_uri: string;
  asset_url: string;
  content_type: string;
  filename: string;
}

// ---------------------------------------------------------------------------
// Interview Questions
// ---------------------------------------------------------------------------

export interface QuestionSetResponse {
  id: string;                            // UUID
  candidate_profile_id: string;         // UUID
  candidate_full_name: string | null;
  job_description_id: string;           // UUID
  job_description_title: string | null;
  generated_by_user_id: string;         // UUID
  question_payload: Record<string, unknown>;
  created_at: string;                    // ISO 8601 datetime
}

export interface QuestionSetListResponse {
  total: number;
  items: QuestionSetResponse[];
}

export interface QuestionSetCreateRequest {
  candidate_profile_id: string;         // UUID
  job_description_id: string;           // UUID
  generated_by_user_id: string;         // UUID
  question_payload: Record<string, unknown>;
}

export interface QuestionSetUpdateRequest {
  question_payload: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Interview Templates / Invitations / Reports
// ---------------------------------------------------------------------------

export interface InterviewTemplateResponse {
  id: string;
  job_id: string;
  name: string;
  language_code: string;
  status: string;
  intro_script: string | null;
  closing_script: string | null;
  question_payload: Record<string, unknown>;
  report_rubric: Record<string, unknown>;
  version: number;
  created_at: string;
  updated_at: string;
}

export interface InterviewTemplateListResponse {
  items: InterviewTemplateResponse[];
  total: number;
}

export interface InterviewTemplateCreateRequest {
  name: string;
  language_code?: string;
  status?: string;
  intro_script?: string | null;
  closing_script?: string | null;
  question_payload?: Record<string, unknown>;
  report_rubric?: Record<string, unknown>;
}

export interface InterviewTemplateUpdateRequest {
  name?: string;
  language_code?: string;
  status?: string;
  intro_script?: string | null;
  closing_script?: string | null;
  question_payload?: Record<string, unknown>;
  report_rubric?: Record<string, unknown>;
}

export interface InterviewInvitationCreateRequest {
  job_id: string;
  candidate_profile_id: string;
  interview_template_id?: string | null;
  interview_question_set_id?: string | null;
  expires_in_hours?: number | null;
  send_email?: boolean;
}

export interface InterviewInvitationResponse {
  id: string;
  job_id: string;
  candidate_profile_id: string;
  candidate_full_name: string | null;
  interview_template_id: string;
  interview_template_name: string | null;
  public_token: string;
  public_url: string;
  status: string;
  expires_at: string | null;
  max_attempts: number;
  attempt_count: number;
  latest_interview_session_id: string | null;
  sent_by_user_id: string | null;
  sent_at: string | null;
  opened_at: string | null;
  completed_at: string | null;
  cancelled_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface InterviewInvitationListResponse {
  items: InterviewInvitationResponse[];
  total: number;
}

export interface DeleteInterviewTemplateResponse {
  deleted: boolean;
  template_id: string;
}

export interface InterviewReportResponse {
  id: string;
  interview_session_id: string;
  interview_template_id: string | null;
  summary_text: string | null;
  report_payload: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface PublicInterviewStartRequest {
  provider?: string | null;
  provider_session_id?: string | null;
  device_metadata?: Record<string, unknown> | null;
  browser_metadata?: Record<string, unknown> | null;
  connection_metadata?: Record<string, unknown> | null;
}

export interface PublicInterviewEventItemRequest {
  speaker: string;
  text: string;
  offset_ms?: number | null;
  question_key?: string | null;
  payload?: Record<string, unknown> | null;
}

export interface PublicInterviewEventsRequest {
  provider?: string | null;
  events: PublicInterviewEventItemRequest[];
}

export interface PublicInterviewCompleteRequest {
  provider?: string | null;
}

export interface PublicInterviewTTSRequest {
  text: string;
}

export interface PublicInterviewInvitationPayload {
  id: string;
  public_token: string;
  status: string;
  expires_at: string | null;
  max_attempts: number;
  attempt_count: number;
  candidate_full_name: string | null;
  completed_at: string | null;
}

export interface PublicInterviewSessionPayload {
  id: string;
  provider: string | null;
  provider_session_id: string | null;
  status: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface PublicInterviewTemplatePayload {
  id: string;
  name: string;
  language_code: string;
  intro_script: string | null;
  closing_script: string | null;
  question_payload: Record<string, unknown>;
}

export interface PublicInterviewAvailabilityPayload {
  can_start: boolean;
  reason: string;
  detail: string | null;
}

export interface PublicInterviewStatusResponse {
  invitation: PublicInterviewInvitationPayload;
  template: PublicInterviewTemplatePayload;
  availability: PublicInterviewAvailabilityPayload;
}

export interface PublicInterviewStartResponse {
  invitation: PublicInterviewInvitationPayload;
  session: PublicInterviewSessionPayload;
  template: PublicInterviewTemplatePayload;
}

export interface PublicInterviewEventsResponse {
  accepted: boolean;
  stored_turns: number;
}

export interface PublicInterviewCompleteResponse {
  invitation: PublicInterviewInvitationPayload;
  session: PublicInterviewSessionPayload;
}

// ---------------------------------------------------------------------------
// Generic delete response shape (upload, job-descriptions)
// ---------------------------------------------------------------------------

export interface DeleteResumeResponse {
  deleted: boolean;
  resume_id: string;
}

export interface DeleteJobDescriptionResponse {
  deleted: boolean;
  job_description_id: string;
}
