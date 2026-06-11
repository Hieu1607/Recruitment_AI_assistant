import { client } from "../client";
import type {
  PublicInterviewStatusResponse,
  PublicInterviewCompleteRequest,
  PublicInterviewCompleteResponse,
  PublicInterviewEventsRequest,
  PublicInterviewEventsResponse,
  PublicInterviewStartRequest,
  PublicInterviewStartResponse,
  PublicInterviewTTSRequest,
} from "../types";

export const interviewPublicApi = {
  async getStatus(token: string): Promise<PublicInterviewStatusResponse> {
    const { data } = await client.get<PublicInterviewStatusResponse>(`/public/interview/${token}`);
    return data;
  },

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

  async synthesizeSpeech(token: string, body: PublicInterviewTTSRequest): Promise<Blob> {
    const { data } = await client.post<Blob>(`/public/interview/${token}/tts`, body, {
      responseType: "blob",
      timeout: 120_000,
    });
    return data;
  },
};
