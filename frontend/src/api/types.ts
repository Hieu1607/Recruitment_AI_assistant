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
  original_file_name: string;
  storage_uri: string;                     // server-side file path
  upload_status: UploadStatus;
  duplicate_group_key: string | null;
  uploaded_by_user_id: string;             // UUID
  uploaded_at: string | null;              // ISO 8601 datetime
  processed_at: string | null;             // ISO 8601 datetime
  retention_expires_at: string | null;     // ISO 8601 datetime
}

export interface ResumeBatchParseItem {
  file_name: string;
  resume_document_id: string;             // UUID
  candidate_profile_id: string | null;    // UUID — null if parsing failed
  status: UploadStatus;
}

export interface ResumeBatchParseResponse {
  total_files: number;
  processed_files: number;
  failed_files: number;
  items: ResumeBatchParseItem[];
}

export interface ResumeListResponse {
  total: number;
  items: ResumeResponse[];
}

// ---------------------------------------------------------------------------
// Job Descriptions
// ---------------------------------------------------------------------------

export interface JobDescriptionResponse {
  id: string;                              // UUID
  title: string | null;
  jd_text: string;
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
  created_by_user_id: string;             // UUID
}

export interface JobDescriptionUpdateRequest {
  title?: string;
  jd_text?: string;
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
  weight: number;
  score: number;
  weightedScore: number;
  evidenceSummary: string;
}

export interface CandidateScore {
  candidateId: string;                   // UUID — camelCase per BACKEND.md note 7
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

export interface ChatResponse {
  session_id: string;
  answer: string;
  candidates_in_scope: number;
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
// Outreach Messages
// ---------------------------------------------------------------------------

export interface OutreachResponse {
  id: string;                            // UUID
  candidate_profile_id: string;         // UUID
  candidate_full_name: string | null;
  created_by_user_id: string;           // UUID
  content_source: ContentSource;
  subject: string;
  body: string;
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
  body: string;
}

export interface OutreachUpdateRequest {
  subject?: string;
  body?: string;
  sent_status?: SentStatus;
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
