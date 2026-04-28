import { client } from "../client";
import type {
  ResumeResponse,
  ResumeListResponse,
  ResumeBatchParseResponse,
  DeleteResumeResponse,
  UploadStatus,
} from "../types";

export const uploadApi = {
  /**
   * List resume documents with optional filters.
   * GET /upload/
   */
  async list(params?: {
    upload_status?: UploadStatus;
    uploaded_by_user_id?: string;
    limit?: number;
    offset?: number;
  }): Promise<ResumeListResponse> {
    const { data } = await client.get<ResumeListResponse>("/upload/", {
      params,
    });
    return data;
  },

  /**
   * Get a single resume document by UUID.
   * GET /upload/{resume_id}
   */
  async get(resumeId: string): Promise<ResumeResponse> {
    const { data } = await client.get<ResumeResponse>(`/upload/${resumeId}`);
    return data;
  },

  /**
   * Update a resume document (filename or status).
   * PATCH /upload/{resume_id}
   */
  async update(
    resumeId: string,
    body: Partial<Pick<ResumeResponse, "original_file_name" | "upload_status">>,
  ): Promise<ResumeResponse> {
    const { data } = await client.patch<ResumeResponse>(
      `/upload/${resumeId}`,
      body,
    );
    return data;
  },

  /**
   * Delete a resume document (and optionally the physical PDF).
   * DELETE /upload/{resume_id}
   * Returns 200 with { deleted, resume_id }.
   */
  async remove(
    resumeId: string,
    deleteFile = false,
  ): Promise<DeleteResumeResponse> {
    const { data } = await client.delete<DeleteResumeResponse>(
      `/upload/${resumeId}`,
      { params: { delete_file: deleteFile } },
    );
    return data;
  },

  /**
   * Upload and parse one or more PDF resumes.
   * POST /upload/batch-parse
   * Processing is synchronous — can take 30+ seconds for large batches.
   */
  async batchParse(
    files: File[],
    uploaded_by_user_id?: string,
  ): Promise<ResumeBatchParseResponse> {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    if (uploaded_by_user_id) {
      fd.append("uploaded_by_user_id", uploaded_by_user_id);
    }
    const { data } = await client.post<ResumeBatchParseResponse>(
      "/upload/batch-parse",
      fd,
      {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 300_000, // 5 min for synchronous LLM batch
      },
    );
    return data;
  },
};
