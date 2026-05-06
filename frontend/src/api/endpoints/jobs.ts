import { client } from "../client";
import type {
  ChatResponse,
  JobDescriptionResponse,
  JobListResponse,
  JobResponse,
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

  async create(body: { title: string; status?: string }): Promise<JobResponse> {
    const { data } = await client.post<JobResponse>("/jobs/", body);
    return data;
  },

  async get(jobId: string): Promise<JobResponse> {
    const { data } = await client.get<JobResponse>(`/jobs/${jobId}`);
    return data;
  },

  async update(jobId: string, body: { title?: string; status?: string }): Promise<JobResponse> {
    const { data } = await client.patch<JobResponse>(`/jobs/${jobId}`, body);
    return data;
  },

  jobDescription: {
    async get(jobId: string): Promise<JobDescriptionResponse> {
      const { data } = await client.get<JobDescriptionResponse>(`/jobs/${jobId}/job-description`);
      return data;
    },
    async upsert(
      jobId: string,
      body: { title?: string; jd_text: string; is_active?: boolean },
    ): Promise<JobDescriptionResponse> {
      const { data } = await client.post<JobDescriptionResponse>(`/jobs/${jobId}/job-description`, body);
      return data;
    },
    async patch(
      jobId: string,
      body: { title?: string; jd_text?: string; is_active?: boolean },
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

  async listCandidates(jobId: string): Promise<{ items: any[]; total: number }> {
    const { data } = await client.get<{ items: any[]; total: number }>(`/jobs/${jobId}/candidates`);
    return data;
  },

  async score(
    jobId: string,
    body: { score_threshold?: number; candidate_profile_ids?: string[]; section_weights?: Record<string, number>; batch_size?: number },
  ): Promise<ScoreResponse> {
    const { data } = await client.post<ScoreResponse>(`/jobs/${jobId}/score`, body);
    return data;
  },

  chat: {
    async send(jobId: string, body: { message: string; session_id?: string; candidate_limit?: number }): Promise<ChatResponse> {
      const { data } = await client.post<ChatResponse>(`/jobs/${jobId}/chat`, body);
      return data;
    },
  },
};
