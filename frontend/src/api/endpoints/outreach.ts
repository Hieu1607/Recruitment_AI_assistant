import { client } from "../client";
import type {
  OutreachAssetUploadResponse,
  OutreachResponse,
  OutreachListResponse,
  OutreachBulkSendRequest,
  OutreachBulkSendResponse,
  OutreachCreateRequest,
  OutreachTemplateCreateRequest,
  OutreachTemplateListResponse,
  OutreachTemplateResponse,
  OutreachTemplateGenerateRequest,
  OutreachTemplateGenerateResponse,
  OutreachTemplateUpdateRequest,
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

  async listTemplates(params?: {
    created_by_user_id?: string;
    job_id?: string;
    limit?: number;
    offset?: number;
  }): Promise<OutreachTemplateListResponse> {
    const { data } = await client.get<OutreachTemplateListResponse>("/outreach/templates", { params });
    return data;
  },

  async createTemplate(body: OutreachTemplateCreateRequest): Promise<OutreachTemplateResponse> {
    const { data } = await client.post<OutreachTemplateResponse>("/outreach/templates", body);
    return data;
  },

  async updateTemplate(
    templateId: string,
    body: OutreachTemplateUpdateRequest,
  ): Promise<OutreachTemplateResponse> {
    const { data } = await client.patch<OutreachTemplateResponse>(`/outreach/templates/${templateId}`, body);
    return data;
  },

  async generateTemplateDraft(
    body: OutreachTemplateGenerateRequest,
  ): Promise<OutreachTemplateGenerateResponse> {
    const { data } = await client.post<OutreachTemplateGenerateResponse>("/outreach/templates/generate-draft", body);
    return data;
  },

  async uploadImage(file: File): Promise<OutreachAssetUploadResponse> {
    const form = new FormData();
    form.append("file", file);
    const { data } = await client.post<OutreachAssetUploadResponse>("/outreach-assets/upload", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return data;
  },

  async send(messageId: string): Promise<OutreachResponse> {
    const { data } = await client.post<OutreachResponse>(
      `/outreach/${messageId}/send`,
    );
    return data;
  },

  async bulkSend(body: OutreachBulkSendRequest): Promise<OutreachBulkSendResponse> {
    const { data } = await client.post<OutreachBulkSendResponse>(
      "/outreach/bulk-send",
      body,
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
