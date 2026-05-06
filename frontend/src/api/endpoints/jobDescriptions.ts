import { client } from "../client";
import type {
  JobDescriptionResponse,
  JobDescriptionListResponse,
  JobDescriptionCreateRequest,
  JobDescriptionUpdateRequest,
  DeleteJobDescriptionResponse,
} from "../types";

export const jobDescriptionsApi = {
  /**
   * List job descriptions with optional filters.
   * GET /job-descriptions/
   */
  async list(params?: {
    is_active?: boolean;
    limit?: number;
    offset?: number;
  }): Promise<JobDescriptionListResponse> {
    const { data } = await client.get<JobDescriptionListResponse>(
      "/job-descriptions/",
      { params },
    );
    return data;
  },

  /**
   * Get a single job description by UUID.
   * GET /job-descriptions/{jd_id}
   */
  async get(jdId: string): Promise<JobDescriptionResponse> {
    const { data } = await client.get<JobDescriptionResponse>(
      `/job-descriptions/${jdId}`,
    );
    return data;
  },

  /**
   * Create a new job description.
   * POST /job-descriptions/
   */
  async create(
    body: JobDescriptionCreateRequest,
  ): Promise<JobDescriptionResponse> {
    const { data } = await client.post<JobDescriptionResponse>(
      "/job-descriptions/",
      body,
    );
    return data;
  },

  /**
   * Partially update a job description.
   * PATCH /job-descriptions/{jd_id}
   */
  async update(
    jdId: string,
    body: JobDescriptionUpdateRequest,
  ): Promise<JobDescriptionResponse> {
    const { data } = await client.patch<JobDescriptionResponse>(
      `/job-descriptions/${jdId}`,
      body,
    );
    return data;
  },

  /**
   * Delete a job description.
   * DELETE /job-descriptions/{jd_id}
   * Returns 200 with { deleted, job_description_id }.
   */
  async remove(jdId: string): Promise<DeleteJobDescriptionResponse> {
    const { data } = await client.delete<DeleteJobDescriptionResponse>(
      `/job-descriptions/${jdId}`,
    );
    return data;
  },
};
