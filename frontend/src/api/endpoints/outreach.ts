import { client } from "../client";
import type {
  OutreachResponse,
  OutreachListResponse,
  OutreachCreateRequest,
  OutreachUpdateRequest,
  SentStatus,
} from "../types";

export const outreachApi = {
  /**
   * List outreach messages with optional filters.
   * GET /outreach/
   * Note: `total` is the real count across all pages (not just the returned page).
   */
  async list(params?: {
    created_by_user_id?: string;
    candidate_profile_id?: string;
    sent_status?: SentStatus;
    limit?: number;
    offset?: number;
  }): Promise<OutreachListResponse> {
    const { data } = await client.get<OutreachListResponse>("/outreach/", {
      params,
    });
    return data;
  },

  /**
   * Get a single outreach message by UUID.
   * GET /outreach/{message_id}
   */
  async get(messageId: string): Promise<OutreachResponse> {
    const { data } = await client.get<OutreachResponse>(
      `/outreach/${messageId}`,
    );
    return data;
  },

  /**
   * Create an outreach message.
   * POST /outreach/
   */
  async create(body: OutreachCreateRequest): Promise<OutreachResponse> {
    const { data } = await client.post<OutreachResponse>("/outreach/", body);
    return data;
  },

  /**
   * Update subject, body, or sent status.
   * PATCH /outreach/{message_id}
   * Setting sent_status to "sent" auto-fills sent_at on the backend.
   */
  async update(
    messageId: string,
    body: OutreachUpdateRequest,
  ): Promise<OutreachResponse> {
    const { data } = await client.patch<OutreachResponse>(
      `/outreach/${messageId}`,
      body,
    );
    return data;
  },

  async send(messageId: string): Promise<OutreachResponse> {
    const { data } = await client.post<OutreachResponse>(
      `/outreach/${messageId}/send`,
    );
    return data;
  },

  /**
   * Delete an outreach message.
   * DELETE /outreach/{message_id} → 204 No Content
   */
  async remove(messageId: string): Promise<void> {
    await client.delete(`/outreach/${messageId}`);
  },
};
