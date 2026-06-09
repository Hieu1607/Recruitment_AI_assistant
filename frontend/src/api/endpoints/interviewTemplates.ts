import { client } from "../client";
import type {
  DeleteInterviewTemplateResponse,
  InterviewTemplateCreateRequest,
  InterviewTemplateListResponse,
  InterviewTemplateResponse,
  InterviewTemplateUpdateRequest,
} from "../types";

export const interviewTemplatesApi = {
  async list(jobId: string): Promise<InterviewTemplateListResponse> {
    const { data } = await client.get<InterviewTemplateListResponse>(
      `/jobs/${jobId}/interview-templates`,
    );
    return data;
  },

  async create(jobId: string, body: InterviewTemplateCreateRequest): Promise<InterviewTemplateResponse> {
    const { data } = await client.post<InterviewTemplateResponse>(
      `/jobs/${jobId}/interview-templates`,
      body,
    );
    return data;
  },

  async get(templateId: string): Promise<InterviewTemplateResponse> {
    const { data } = await client.get<InterviewTemplateResponse>(`/interview-templates/${templateId}`);
    return data;
  },

  async update(
    templateId: string,
    body: InterviewTemplateUpdateRequest,
  ): Promise<InterviewTemplateResponse> {
    const { data } = await client.patch<InterviewTemplateResponse>(
      `/interview-templates/${templateId}`,
      body,
    );
    return data;
  },

  async remove(templateId: string): Promise<DeleteInterviewTemplateResponse> {
    const { data } = await client.delete<DeleteInterviewTemplateResponse>(
      `/interview-templates/${templateId}`,
    );
    return data;
  },
};
