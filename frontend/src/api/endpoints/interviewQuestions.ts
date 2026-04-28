import { client } from "../client";
import type {
  QuestionSetResponse,
  QuestionSetListResponse,
  QuestionSetCreateRequest,
  QuestionSetUpdateRequest,
} from "../types";

export const interviewQuestionsApi = {
  /**
   * List interview question sets with optional filters.
   * GET /interview-questions/
   */
  async list(params?: {
    generated_by_user_id?: string;
    candidate_profile_id?: string;
    job_description_id?: string;
    limit?: number;
    offset?: number;
  }): Promise<QuestionSetListResponse> {
    const { data } = await client.get<QuestionSetListResponse>(
      "/interview-questions/",
      { params },
    );
    return data;
  },

  /**
   * Get a single interview question set by UUID.
   * GET /interview-questions/{question_set_id}
   */
  async get(questionSetId: string): Promise<QuestionSetResponse> {
    const { data } = await client.get<QuestionSetResponse>(
      `/interview-questions/${questionSetId}`,
    );
    return data;
  },

  /**
   * Create a new interview question set.
   * POST /interview-questions/
   */
  async create(body: QuestionSetCreateRequest): Promise<QuestionSetResponse> {
    const { data } = await client.post<QuestionSetResponse>(
      "/interview-questions/",
      body,
    );
    return data;
  },

  /**
   * Replace the question payload for an existing set.
   * PATCH /interview-questions/{question_set_id}
   */
  async update(
    questionSetId: string,
    body: QuestionSetUpdateRequest,
  ): Promise<QuestionSetResponse> {
    const { data } = await client.patch<QuestionSetResponse>(
      `/interview-questions/${questionSetId}`,
      body,
    );
    return data;
  },

  /**
   * Delete an interview question set.
   * DELETE /interview-questions/{question_set_id} → 204 No Content
   */
  async remove(questionSetId: string): Promise<void> {
    await client.delete(`/interview-questions/${questionSetId}`);
  },
};
