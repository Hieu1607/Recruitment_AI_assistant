import { client } from "../client";
import type {
  InterviewInvitationCreateRequest,
  InterviewInvitationListResponse,
  InterviewInvitationResponse,
} from "../types";

export const interviewInvitationsApi = {
  async create(body: InterviewInvitationCreateRequest): Promise<InterviewInvitationResponse> {
    const { data } = await client.post<InterviewInvitationResponse>("/interview-invitations", body);
    return data;
  },

  async list(jobId: string): Promise<InterviewInvitationListResponse> {
    const { data } = await client.get<InterviewInvitationListResponse>(
      `/jobs/${jobId}/interview-invitations`,
    );
    return data;
  },

  async revoke(invitationId: string): Promise<InterviewInvitationResponse> {
    const { data } = await client.post<InterviewInvitationResponse>(
      `/interview-invitations/${invitationId}/revoke`,
    );
    return data;
  },
};
