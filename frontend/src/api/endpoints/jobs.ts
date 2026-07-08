import { client } from "../client";
import type {
  CandidateProfileResponse,
  ChatResponse,
  CandidateEvaluationResponse,
  ChatSessionListResponse,
  ChatSessionResponse,
  ChatTurnResponse,
  JobDescriptionResponse,
  JobEvaluationListResponse,
  JobApplicationLinkResponse,
  JobListResponse,
  JobScoringPreferenceResponse,
  JobResponse,
  JobSetupStatusResponse,
  ResumeBatchParseResponse,
  ResumeListResponse,
  ResumeResponse,
  ScoreResponse,
} from "../types";

export const jobsApi = {
  async list(): Promise<JobListResponse> {
    const { data } = await client.get<JobListResponse>("/jobs/");
    return data;
  },

  async create(body: {
    title: string;
    status?: string;
    candidate_message?: string | null;
    public_apply_enabled?: boolean;
  }): Promise<JobResponse> {
    const { data } = await client.post<JobResponse>("/jobs/", body);
    return data;
  },

  async get(jobId: string): Promise<JobResponse> {
    const { data } = await client.get<JobResponse>(`/jobs/${jobId}`);
    return data;
  },

  async update(jobId: string, body: {
    title?: string;
    status?: string;
    candidate_message?: string | null;
    public_apply_enabled?: boolean;
  }): Promise<JobResponse> {
    const { data } = await client.patch<JobResponse>(`/jobs/${jobId}`, body);
    return data;
  },

  async remove(jobId: string): Promise<{ deleted: boolean; job_id: string }> {
    const { data } = await client.delete<{ deleted: boolean; job_id: string }>(`/jobs/${jobId}`);
    return data;
  },

  applicationLink: {
    async get(jobId: string): Promise<JobApplicationLinkResponse> {
      const { data } = await client.get<JobApplicationLinkResponse>(`/jobs/${jobId}/application-link`);
      return data;
    },
    async rotate(jobId: string): Promise<JobApplicationLinkResponse> {
      const { data } = await client.post<JobApplicationLinkResponse>(`/jobs/${jobId}/application-link/rotate`);
      return data;
    },
  },

  jobDescription: {
    async get(jobId: string): Promise<JobDescriptionResponse> {
      const { data } = await client.get<JobDescriptionResponse>(`/jobs/${jobId}/job-description`);
      return data;
    },
    async upsert(
      jobId: string,
      body: { title?: string; jd_text: string; hidden_text?: string; is_active?: boolean },
    ): Promise<JobDescriptionResponse> {
      const { data } = await client.post<JobDescriptionResponse>(`/jobs/${jobId}/job-description`, body);
      return data;
    },
    async patch(
      jobId: string,
      body: { title?: string; jd_text?: string; hidden_text?: string; is_active?: boolean },
    ): Promise<JobDescriptionResponse> {
      const { data } = await client.patch<JobDescriptionResponse>(`/jobs/${jobId}/job-description`, body);
      return data;
    },
  },

  resumes: {
    async list(jobId: string, params?: { upload_status?: string; limit?: number; offset?: number }): Promise<ResumeListResponse> {
      const { data } = await client.get<ResumeListResponse>(`/jobs/${jobId}/resumes`, { params });
      return data;
    },
    async get(jobId: string, resumeId: string): Promise<ResumeResponse> {
      const { data } = await client.get<ResumeResponse>(`/jobs/${jobId}/resumes/${resumeId}`);
      return data;
    },
    async update(jobId: string, resumeId: string, body: Partial<Pick<ResumeResponse, "original_file_name" | "upload_status">>): Promise<ResumeResponse> {
      const { data } = await client.patch<ResumeResponse>(`/jobs/${jobId}/resumes/${resumeId}`, body);
      return data;
    },
    async remove(jobId: string, resumeId: string): Promise<{ deleted: boolean; resume_id: string }> {
      const { data } = await client.delete<{ deleted: boolean; resume_id: string }>(`/jobs/${jobId}/resumes/${resumeId}`);
      return data;
    },
    async batchParse(jobId: string, files: File[]): Promise<ResumeBatchParseResponse> {
      const fd = new FormData();
      files.forEach((file) => fd.append("files", file));

      const { data } = await client.post<ResumeBatchParseResponse>(`/jobs/${jobId}/resumes`, fd, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 300_000,
      });
      return data;
    },
  },

  async listCandidates(jobId: string): Promise<{ items: CandidateProfileResponse[]; total: number }> {
    const { data } = await client.get<{ items: CandidateProfileResponse[]; total: number }>(`/jobs/${jobId}/candidates`);
    return data;
  },

  async score(
    jobId: string,
    body: { score_threshold?: number; candidate_profile_ids?: string[]; section_weights?: Record<string, number>; batch_size?: number },
  ): Promise<ScoreResponse> {
    const { data } = await client.post<ScoreResponse>(`/jobs/${jobId}/score`, body);
    return data;
  },

  scoreRuns: {
    async get(jobId: string, matchRunId: string): Promise<ScoreResponse> {
      const { data } = await client.get<ScoreResponse>(`/jobs/${jobId}/score-runs/${matchRunId}`);
      return data;
    },
  },

  evaluations: {
    async list(jobId: string): Promise<JobEvaluationListResponse> {
      const { data } = await client.get<JobEvaluationListResponse>(`/jobs/${jobId}/evaluations`);
      return data;
    },
    async getCandidate(jobId: string, candidateProfileId: string): Promise<CandidateEvaluationResponse> {
      const { data } = await client.get<CandidateEvaluationResponse>(`/jobs/${jobId}/candidates/${candidateProfileId}/evaluation`);
      return data;
    },
    async scoreAgain(jobId: string): Promise<{ queued: number; total_candidates: number }> {
      const { data } = await client.post<{ queued: number; total_candidates: number }>(`/jobs/${jobId}/evaluations/score-again`);
      return data;
    },
  },

  scoringPreferences: {
    async update(
      jobId: string,
      body: { section_weights: Record<string, number>; score_threshold: number },
    ): Promise<JobScoringPreferenceResponse> {
      const { data } = await client.put<JobScoringPreferenceResponse>(`/jobs/${jobId}/scoring-preferences`, body);
      return data;
    },
  },

  chat: {
    async send(jobId: string, body: { message: string; session_id?: string; candidate_limit?: number }): Promise<ChatResponse> {
      const { data } = await client.post<ChatResponse>(`/jobs/${jobId}/chat`, body);
      return data;
    },

    sessions: {
      async list(jobId: string, params?: { limit?: number; offset?: number }): Promise<ChatSessionListResponse> {
        const { data } = await client.get<ChatSessionListResponse>(`/jobs/${jobId}/chat/sessions`, { params });
        return data;
      },
      async create(jobId: string, body?: { session_title?: string | null }): Promise<ChatSessionResponse> {
        const { data } = await client.post<ChatSessionResponse>(`/jobs/${jobId}/chat/sessions`, body ?? {});
        return data;
      },
      async get(jobId: string, sessionId: string): Promise<ChatSessionResponse> {
        const { data } = await client.get<ChatSessionResponse>(`/jobs/${jobId}/chat/sessions/${sessionId}`);
        return data;
      },
      async update(jobId: string, sessionId: string, body: { session_title?: string | null }): Promise<ChatSessionResponse> {
        const { data } = await client.patch<ChatSessionResponse>(`/jobs/${jobId}/chat/sessions/${sessionId}`, body);
        return data;
      },
      async remove(jobId: string, sessionId: string): Promise<void> {
        await client.delete(`/jobs/${jobId}/chat/sessions/${sessionId}`);
      },
    },

    turns: {
      async list(jobId: string, sessionId: string, params?: { limit?: number; offset?: number }): Promise<ChatTurnResponse[]> {
        const { data } = await client.get<ChatTurnResponse[]>(`/jobs/${jobId}/chat/sessions/${sessionId}/turns`, { params });
        return data;
      },
    },
  },

  setupStatus: {
    async get(jobId: string): Promise<JobSetupStatusResponse> {
      const { data } = await client.get<JobSetupStatusResponse>(`/jobs/${jobId}/setup-status`);
      return data;
    },
  },
};
