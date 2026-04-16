// --- Pagination ---
export interface PaginatedResponse<T> {
  total: number;
  items: T[];
}

// --- Resumes ---
export interface ResumeResponse {
  id: string;
  original_file_name: string;
  storage_uri: string;
  upload_status: 'uploaded' | 'processing' | 'processed' | 'failed';
  duplicate_group_key: string | null;
  uploaded_by_user_id: string;
  uploaded_at: string | null;
  processed_at: string | null;
  retention_expires_at: string | null;
}

export interface BatchParseResponse {
  total_files: number;
  processed_files: number;
  failed_files: number;
  items: {
    file_name: string;
    resume_document_id: string;
    candidate_profile_id: string;
    status: string;
  }[];
}

// --- Job Descriptions ---
export interface JobDescriptionResponse {
  id: string;
  title: string | null;
  jd_text: string;
  created_by_user_id: string;
  created_at: string;
  is_active: boolean;
}

// --- Scoring ---
export interface ComponentScore {
  criterionKey: string;
  weight: number;
  score: number;
  weightedScore: number;
  evidenceSummary: string;
}

export interface CandidateScore {
  candidateId: string;
  totalScore: number;
  passedThreshold: boolean;
  rationale: string;
  componentScores: ComponentScore[];
}

export interface ScoreResponse {
  match_run_id: string;
  job_description_id: string;
  total_candidates: number;
  total_passed_candidates: number;
  batches: number;
  scores: CandidateScore[];
}

// --- Chat ---
export interface ChatResponse {
  session_id: string;
  answer: string;
  candidates_in_scope: number;
}

export interface ChatMessage {
  role: 'human' | 'ai';
  content: string;
}

export interface ChatHistoryResponse {
  session_id: string;
  messages: ChatMessage[];
}

// --- Shortlist ---
export interface SessionResponse {
  id: string;
  user_id: string;
  session_title: string | null;
  turn_count: number;
  created_at: string;
  updated_at: string;
}

export interface TurnResponse {
  id: string;
  query_session_id: string;
  user_question: string;
  answer_text: string;
  matched_candidate_ids: string[] | null;
  matched_count: number | null;
  tool_trace_masked: Record<string, unknown> | null;
  created_at: string;
}

export interface CollectionResponse {
  id: string;
  name: string;
  created_by_user_id: string;
  source_query_turn_id: string | null;
  item_count: number;
  created_at: string;
}

export interface ShortlistItemResponse {
  id: string;
  shortlist_collection_id: string;
  candidate_profile_id: string;
  added_at: string;
}

// --- Outreach ---
export interface OutreachResponse {
  id: string;
  candidate_profile_id: string;
  candidate_full_name: string | null;
  created_by_user_id: string;
  content_source: 'ai_draft' | 'template';
  subject: string;
  body: string;
  sent_status: 'not_sent' | 'sent' | 'failed';
  sent_at: string | null;
  created_at: string;
}

// --- Interview Questions ---
export interface QuestionSetResponse {
  id: string;
  candidate_profile_id: string;
  candidate_full_name: string | null;
  job_description_id: string;
  job_description_title: string | null;
  generated_by_user_id: string;
  question_payload: Record<string, unknown>;
  created_at: string;
}
