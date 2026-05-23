import { client } from "../client";
import type {
  PublicInterviewCompleteRequest,
  PublicInterviewCompleteResponse,
  PublicInterviewEventsRequest,
  PublicInterviewEventsResponse,
  PublicInterviewStartRequest,
  PublicInterviewStartResponse,
} from "../types";

export const interviewPublicApi = {
  async start(token: string, body: PublicInterviewStartRequest): Promise<PublicInterviewStartResponse> {
    const { data } = await client.post<PublicInterviewStartResponse>(`/public/interview/${token}/start`, body);
    return data;
  },

  async ingestEvents(token: string, body: PublicInterviewEventsRequest): Promise<PublicInterviewEventsResponse> {
    const { data } = await client.post<PublicInterviewEventsResponse>(`/public/interview/${token}/events`, body);
    return data;
  },

  async complete(token: string, body: PublicInterviewCompleteRequest): Promise<PublicInterviewCompleteResponse> {
    const { data } = await client.post<PublicInterviewCompleteResponse>(`/public/interview/${token}/complete`, body);
    return data;
  },
};
